"""Build a deterministic review queue for disputed predicted facts.

This is a product-side aid for selective verification by a stronger model or a
human. It reads only the input calls and the model predictions; gold facts are
stripped at the ``as_call`` boundary and never reach the heuristics. No teacher
API is called here.

The queue is intentionally heuristic: it ranks facts that are hard to support
from the transcript, have an unsupported date, compete for one current state,
or look like an offer/conditional statement. It does not decide whether a fact
is wrong.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import re

from case_io import read_jsonl, write_jsonl
from schema import as_call
from temporal import _cue_dates, extract_time_expressions

WORD = re.compile(r'[а-яa-z0-9+]+')
STOPWORDS = {
    'а', 'бы', 'был', 'была', 'были', 'было', 'быть', 'в', 'вам', 'вас', 'весь',
    'все', 'всё', 'вы', 'где', 'да', 'для', 'до', 'его', 'ее', 'её', 'если', 'есть',
    'еще', 'ещё', 'же', 'за', 'здесь', 'и', 'из', 'или', 'им', 'их', 'к', 'как',
    'какая', 'какие', 'какой', 'когда', 'кто', 'ли', 'мне', 'может', 'можно', 'мы',
    'на', 'над', 'надо', 'нам', 'нас', 'не', 'него', 'нее', 'неё', 'нет', 'ни',
    'но', 'ну', 'о', 'об', 'он', 'она', 'они', 'оно', 'от', 'по', 'при', 'с',
    'сам', 'сейчас', 'так', 'там', 'те', 'то', 'тоже', 'только', 'тут', 'у',
    'уже', 'что', 'чтобы', 'это', 'этот', 'эта', 'эти', 'я',
}
SINGLE_VALUED = {'service_state', 'tariff', 'ticket_status'}
OFFER = re.compile(r'\b(?:если|возможно|наверное|кажется|может|предлага|предлож|'
                   r'хотите|хотит|давайте|рассмотр|интересу|план|завтра|будет|будут)')
DATE_CUE = re.compile(r'\b(?:вчера|позавчера|сегодня|завтра|послезавтра|'
                      r'через|назад|недел|месяц|год|числ|'
                      r'понедельник|вторник|сред[аыу]|четверг|пятниц|суббот|воскресень)')


def _words(text):
    return [word for word in WORD.findall((text or '').lower().replace('ё', 'е'))
            if word not in STOPWORDS]


def _stem(word):
    if len(word) >= 7:
        return word[:-2]
    if len(word) >= 5:
        return word[:-1]
    return word


def _value_support(value, source_text):
    value_tokens = [_stem(word) for word in _words(value)]
    if not value_tokens:
        return 0.0
    source_tokens = {_stem(word) for word in _words(source_text)}
    supported = sum(1 for token in value_tokens if token in source_tokens)
    return supported / len(value_tokens)


def _object_key(obj):
    obj = obj or {}
    if obj.get('id'):
        return ('id', obj['id'])
    name = ' '.join(WORD.findall((obj.get('name') or '').lower().replace('ё', 'е')))
    return ('name', name, obj.get('type'))


def _source_has_date(text, started_at):
    if not text:
        return False
    try:
        if extract_time_expressions(text, started_at, direction='past'):
            return True
        anchor = datetime.fromisoformat(started_at.replace('Z', '+00:00')).date()
        return bool(_cue_dates(text, anchor, fuzzy=True))
    except (TypeError, ValueError):
        return bool(DATE_CUE.search(text.lower().replace('ё', 'е')))


def _turn_window(call, turn_idx):
    turns = sorted(call.turns, key=lambda turn: turn.idx)
    by_idx = {turn.idx: turn for turn in turns}
    turn = by_idx.get(turn_idx)
    if turn is None:
        return None
    position = turns.index(turn)
    previous = turns[position - 1] if position else None
    following = turns[position + 1] if position + 1 < len(turns) else None
    return {
        'turn_idx': turn.idx,
        'speaker': turn.speaker,
        'text': turn.text,
        'previous': ({'turn_idx': previous.idx, 'speaker': previous.speaker,
                      'text': previous.text} if previous else None),
        'next': ({'turn_idx': following.idx, 'speaker': following.speaker,
                  'text': following.text} if following else None),
    }


def _candidate_turns(call, fact, current_idx, limit=3):
    """Top turns by lexical support; the current source is marked, never replaced."""
    value = fact.get('value') or ''
    candidates = []
    for turn in sorted(call.turns, key=lambda item: item.idx):
        candidates.append({
            'turn_idx': turn.idx,
            'speaker': turn.speaker,
            'support': round(_value_support(value, turn.text), 4),
            'is_current': turn.idx == current_idx,
            'text': turn.text,
        })
    candidates.sort(key=lambda item: (-item['support'], 0 if item['is_current'] else 1,
                                      item['turn_idx']))
    return candidates[:limit]


def build_queue(calls, predictions, *, support_threshold=0.5, long_agreement_tokens=8,
                max_items=0):
    by_id = {call.call_id: call for call in calls}
    seen = set()
    queued = []
    stats = Counter()
    for prediction in predictions:
        call_id = prediction.get('call_id')
        if call_id not in by_id:
            stats['unknown_call_ids'] += 1
            continue
        if call_id in seen:
            raise ValueError(f'duplicate prediction call_id: {call_id}')
        seen.add(call_id)
        if prediction.get('error'):
            stats['failed_calls'] += 1
            continue
        call = by_id[call_id]
        facts = prediction.get('facts') or []
        stats['facts'] += len(facts)
        reasons_by_index = defaultdict(list)
        support_by_index = {}
        current_groups = defaultdict(list)
        for index, fact in enumerate(facts):
            source = fact.get('source') or {}
            turn = _turn_window(call, source.get('turn_idx'))
            value = fact.get('value') or ''
            support = _value_support(value, turn['text'] if turn else '')
            support_by_index[index] = support
            if turn is None:
                reasons_by_index[index].append('source_turn_missing')
            else:
                if fact.get('speaker') and fact['speaker'] != turn['speaker']:
                    reasons_by_index[index].append('source_speaker_mismatch')
                if support < support_threshold:
                    reasons_by_index[index].append('low_value_support')
                if fact.get('attribute') == 'agreement' and len(_words(value)) > long_agreement_tokens:
                    reasons_by_index[index].append('long_agreement')
                if fact.get('attribute') in {'tariff', 'agreement'} and OFFER.search(
                        turn['text'].lower().replace('ё', 'е')):
                    reasons_by_index[index].append('possible_offer_or_hypothesis')
                if fact.get('valid_from') or fact.get('valid_to'):
                    if not _source_has_date(turn['text'], call.started_at):
                        started_day = call.started_at[:10]
                        from_day = (fact.get('valid_from') or '')[:10]
                        if fact.get('valid_to') or from_day != started_day:
                            reasons_by_index[index].append('date_not_supported_by_source')
            if fact.get('attribute') in SINGLE_VALUED and fact.get('valid_to') is None:
                current_groups[(_object_key(fact.get('object')), fact['attribute'])].append(index)
        for indexes in current_groups.values():
            values = {str(facts[index].get('value') or '').strip().lower() for index in indexes}
            if len(values) > 1:
                for index in indexes:
                    reasons_by_index[index].append('competing_current_values')
        for index, reasons in reasons_by_index.items():
            if not reasons:
                continue
            reasons = sorted(set(reasons))
            priority = 3 if {'source_turn_missing', 'source_speaker_mismatch',
                              'competing_current_values'} & set(reasons) else (
                2 if {'date_not_supported_by_source', 'low_value_support',
                      'possible_offer_or_hypothesis'} & set(reasons) else 1)
            fact = facts[index]
            source_idx = (fact.get('source') or {}).get('turn_idx')
            source_reasons = {'source_turn_missing', 'source_speaker_mismatch',
                              'low_value_support'}
            item = {
                'call_id': call_id,
                'fact_index': index,
                'priority': priority,
                'reasons': reasons,
                'value_support': round(support_by_index[index], 4),
                'attribute': fact.get('attribute'),
                'fact': fact,
                'source': _turn_window(call, source_idx),
                'candidate_turns': (_candidate_turns(call, fact, source_idx)
                                    if source_reasons & set(reasons) else []),
                'call_started_at': call.started_at,
            }
            queued.append(item)
            stats['queued_facts'] += 1
            for reason in reasons:
                stats['reason:' + reason] += 1
            stats['priority:' + str(priority)] += 1
            stats['attribute:' + str(fact.get('attribute'))] += 1
        if reasons_by_index:
            stats['calls_with_queue'] += 1
    queued.sort(key=lambda item: (-item['priority'], item['value_support'],
                                  -len(item['reasons']), item['call_id'], item['fact_index']))
    all_queued = list(queued)
    if max_items:
        queued = queued[:max_items]
    summary = {
        'policy': 'no-gold-heuristic-v1',
        'support_threshold': support_threshold,
        'long_agreement_tokens': long_agreement_tokens,
        'max_items': max_items or None,
        'selection_policy': ('top by priority, then lowest value support, then reason count'
                             if max_items else 'all queued facts'),
        'calls': len(calls),
        'predictions': len(predictions),
        'facts': stats['facts'],
        'queued_facts': len(all_queued),
        'queued_share': round(len(all_queued) / stats['facts'], 4) if stats['facts'] else 0.0,
        'selected_facts': len(queued),
        'selected_share': round(len(queued) / stats['facts'], 4) if stats['facts'] else 0.0,
        'calls_with_queue': stats['calls_with_queue'],
        'failed_calls': stats['failed_calls'],
        'unknown_call_ids': stats['unknown_call_ids'],
        'by_reason': {key[7:]: value for key, value in sorted(stats.items()) if key.startswith('reason:')},
        'by_attribute': {key[10:]: value for key, value in sorted(stats.items()) if key.startswith('attribute:')},
        'by_priority': {key[9:]: value for key, value in sorted(stats.items()) if key.startswith('priority:')},
    }
    return queued, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='calls with turns; gold facts are stripped')
    parser.add_argument('--pred', required=True, help='predictions JSONL')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--support-threshold', type=float, default=0.5)
    parser.add_argument('--long-agreement-tokens', type=int, default=8)
    parser.add_argument('--max-items', type=int, default=0,
                        help='0 = write all queued facts; N = write top N for review')
    args = parser.parse_args()
    if not 0 <= args.support_threshold <= 1:
        parser.error('support threshold must be within [0, 1]')
    if args.long_agreement_tokens < 1:
        parser.error('long agreement threshold must be positive')
    if args.max_items < 0:
        parser.error('max items must be zero or positive')
    output = Path(args.output_dir)
    if output.exists():
        parser.error('Output directory already exists; choose a fresh one')
    calls = [as_call(row) for row in read_jsonl(args.input)]
    predictions = read_jsonl(args.pred)
    queued, summary = build_queue(
        calls, predictions, support_threshold=args.support_threshold,
        long_agreement_tokens=args.long_agreement_tokens, max_items=args.max_items)
    output.mkdir(parents=True)
    write_jsonl(output / 'queue.jsonl', queued)
    with (output / 'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
