"""Shared tariff evidence + verified numbered-ticket assertions. No labels."""
import copy
from datetime import datetime
import os
import tariff_recovery
from ticket_recovery import reconcile_ticket_statuses

TARIFF_BOUNDARY_POLICY=os.getenv('TARIFF_BOUNDARY_POLICY','explicit')
RECOVERY_POLICY='unified-v1-guarded-tariffs-'+TARIFF_BOUNDARY_POLICY


def recover_facts(facts,call,catalog_path=None):
    if TARIFF_BOUNDARY_POLICY not in ('explicit','observation'):
        raise ValueError('Unknown tariff boundary policy')
    original=copy.deepcopy(facts)
    output,decisions=tariff_recovery.recover_facts(facts,call,catalog_path)
    extras=output[len(original):]
    logs=[]
    if extras:
        from state_intervals import bind_explicit_interval
        from temporal_case_v2 import _iso
        entries=[value for value,_pattern in tariff_recovery._load_tariffs(catalog_path)]
        anchor=datetime.fromisoformat(call.started_at.replace('Z','+00:00'))
        turns={t.idx:t for t in call.turns}
        for fresh in extras:
            turn=turns[fresh['source']['turn_idx']]
            changes=[]
            evidence=bind_explicit_interval(fresh,turn,anchor.date(),entries,changes,
                                           lambda d:_iso(d,anchor.tzinfo))
            explicit=any(e['field']=='valid_from' for e in evidence)
            related=[]
            for before,after in zip(original,output):
                if before.get('attribute')=='tariff' and before.get('valid_to')!=after.get('valid_to'):
                    related.append((before,after))
            cannot_close=fresh.get('valid_to') is not None
            removals=[]
            for before,after in related:
                source=turns.get(before['source']['turn_idx'])
                old_names={n for n,_h in tariff_recovery._tariff_hits(before['value'],tariff_recovery._load_tariffs(catalog_path))[1]}
                said={n for n,_h in tariff_recovery._tariff_hits(source.text if source else '',tariff_recovery._load_tariffs(catalog_path))[1]}
                unsupported=bool(old_names) and not old_names.intersection(said) and source \
                    and source.idx<=turn.idx
                if unsupported:
                    removals.append(after)
                    logs.append({'component':'fact_recovery','field':'tariff','action':'remove',
                        'reason':'unsupported_current_tariff','before':before,'source_turn_idx':turn.idx})
                elif explicit:
                    try:
                        valid=not before.get('valid_from') or datetime.fromisoformat(before['valid_from'].replace('Z','+00:00'))<=datetime.fromisoformat(fresh['valid_from'].replace('Z','+00:00'))
                    except (ValueError,TypeError):valid=False
                    if valid:after['valid_to']=fresh['valid_from']
                    else:cannot_close=True
                    logs.append({'component':'fact_recovery','field':'tariff','action':'close',
                        'reason':'explicit_tariff_transition_boundary','before':before,'after':copy.deepcopy(after),'source_turn_idx':turn.idx})
                elif TARIFF_BOUNDARY_POLICY=='explicit':
                    cannot_close=True
            if cannot_close:
                output=copy.deepcopy(original)
                decisions=[];logs=[{'component':'fact_recovery','field':'tariff','action':'abstained',
                    'reason':'tariff_transition_boundary_unknown_or_invalid','source_turn_idx':turn.idx}]
                break
            for old in removals:output.remove(old)
            if evidence:logs.append({'component':'fact_recovery','field':'tariff','action':'date',
                    'reason':'recovered_tariff_explicit_dates','changes':changes,'evidence':evidence,'source_turn_idx':turn.idx})
    output,status_logs=reconcile_ticket_statuses(output,call)
    all_logs=decisions+logs+status_logs
    for log in all_logs:log['policy']=RECOVERY_POLICY
    return output,all_logs
