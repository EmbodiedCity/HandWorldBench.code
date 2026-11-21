"""Collect existing results without starting inference."""
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from handworldbench.runtime.storage import read,write
from handworldbench.runtime.queue import connect
from handworldbench.registry import METRICS,CATEGORIES

REPORT_CATEGORIES = ('all', 'human', 'handson', 'real_robot')
BENCHMARK_CASES = 500  # used only for the completeness warning below; never affects scores


def collect(run):
    run=Path(run);slots=read(run/'configs/slots.json')
    with connect(run) as c:jobs={r['id']:dict(r) for r in c.execute('SELECT * FROM jobs')}
    records=[]
    for slot in slots:
        record=copy.deepcopy(slot);job=jobs.get(slot.get('task_id'))
        if job:
            response=json.loads(job['result']) if job['result'] else {}
            if job['state']=='done':record.update(response['value'])
            elif job['state'] in ('failed','blocked'):
                record.update(status='input_binding_error' if response.get('status')=='input_binding_error' else 'execution_error',score_0_1=None,raw_score=None,failure_reason=response.get('reason') or 'Required task failed')
            else:record.update(status=job['state'],score_0_1=None,raw_score=None)
        record.update(run_dir=str(run.resolve()),metric_name=METRICS[record['metric_id']].name,category_id=METRICS[record['metric_id']].category)
        score=record.get('score_0_1')
        if score is not None and (not math.isfinite(score) or not 0<=score<=1):raise ValueError('Result is outside normalized score range')
        records.append(record)
    return records


def _hwbscore(records, summary, models, selected_metrics):
    """Aggregate six group means from the native, protocol-scoped common scores."""
    metric_rows = defaultdict(list)
    for record in records:
        metric = record['metric_id']
        metric_rows['all', metric].append(record)
        metric_rows[record['category'], metric].append(record)
    summary_scores = {
        (row['model'], row['category'], row['metric_id'], row['metric_protocol']):
            row['score_0_1']
        for row in summary
    }
    metrics_by_group = {
        group: [metric for metric, spec in METRICS.items() if spec.category == group]
        for group in CATEGORIES
    }

    complete_selection = selected_metrics is None or selected_metrics == set(METRICS)
    totals = []
    for category in REPORT_CATEGORIES:
        if not any((category, metric) in metric_rows for metric in METRICS):
            continue
        not_applicable = []
        protocols = {}
        for metric in METRICS:
            rows = metric_rows.get((category, metric), [])
            applicable_rows = [row for row in rows
                               if row.get('applicability', row['status']) != 'not_applicable']
            if rows and not applicable_rows:
                not_applicable.append(metric)
                continue
            protocols[metric] = {row['metric_protocol'] for row in applicable_rows}
        mixed_protocols = [metric for metric, versions in protocols.items() if len(versions) > 1]

        for model in models:
            metric_scores = {
                metric: summary_scores.get((model, category, metric, next(iter(versions))))
                if len(versions) == 1 else None
                for metric, versions in protocols.items()
            }
            missing_metrics = [metric for metric, score in metric_scores.items()
                               if score is None and metric not in mixed_protocols
                               and (selected_metrics is None or metric in selected_metrics)]
            group_scores = {}
            for group, metrics_in_group in metrics_by_group.items():
                values = [metric_scores[metric] for metric in metrics_in_group if metric in protocols]
                group_scores[group] = (math.fsum(values) / len(values)
                                       if values and all(value is not None for value in values)
                                       else None)

            if not complete_selection:
                status = 'incomplete_metric_selection'
            elif mixed_protocols:
                status = 'mixed_protocols'
            elif missing_metrics:
                status = 'unavailable_common_metrics'
            elif any(value is None for value in group_scores.values()):
                status = 'empty_metric_group'
            else:
                status = 'ok'
            total = (100 * math.fsum(group_scores.values()) / len(group_scores)
                     if status == 'ok' else None)
            totals.append(dict(
                model=model, category=category, HWBScore=total, status=status,
                group_scores=group_scores, not_applicable_metrics=not_applicable,
                missing_common_metrics=missing_metrics, mixed_protocol_metrics=mixed_protocols,
            ))
    return totals


def report(runs,output=None,case_ids=None,metrics=None):
    records=[r for run in runs for r in collect(run)];unique={}
    if case_ids is not None:
        case_ids=set(case_ids);records=[r for r in records if r['case_id'] in case_ids]
    if metrics is not None:
        metrics=set(metrics);records=[r for r in records if r['metric_id'] in metrics]
    for r in records:
        key=(r['model'],r['case_id'],r['metric_id'],r['metric_protocol'])
        if key in unique:raise ValueError('Duplicate model/case/protocol in supplied runs')
        unique[key]=r
    models=sorted({r['model'] for r in records});summary=[]
    for metric in METRICS:
        collection=METRICS[metric].level=='collection'
        for category in REPORT_CATEGORIES:
            subset=[r for r in records if r['metric_id']==metric and (category=='all' or r['category']==category)]
            for protocol in sorted({r['metric_protocol'] for r in subset}):
                rows=[r for r in subset if r['metric_protocol']==protocol]
                def valid(r):return bool(r.get('features')) if collection else r.get('score_0_1') is not None
                sets=[{r['case_id'] for r in rows if r['model']==b and valid(r)} for b in models]
                common=set.intersection(*sets) if sets else set()
                for model in models:
                    selected=[r for r in rows if r['model']==model];scored=[r for r in selected if r['case_id'] in common and valid(r)]
                    statuses={s:sum(r['status']==s for r in selected) for s in sorted({r['status'] for r in selected})}
                    value=dict(score_0_1=None)
                    if collection and scored:
                        from handworldbench.metrics.distribution import aggregate
                        value=aggregate(sorted(scored,key=lambda r:r['case_id']))
                    elif scored:
                        value['score_0_1']=sum(r['score_0_1'] for r in scored)/len(scored)
                        raw=[r.get('raw_score') for r in scored]
                        units={r.get('raw_unit') for r in scored}
                        if len(units)==1 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in raw):
                            value['raw_score']=sum(raw)/len(raw)
                            value['raw_unit']=next(iter(units))
                    summary.append(dict(model=model,metric_id=metric,metric_protocol=protocol,category=category,common_cases=len(scored),common_case_ids=sorted(common),status_counts=statuses,**value))
    folder=Path(output) if output else Path(runs[0])/'reports'
    folder.mkdir(parents=True,exist_ok=True);write(folder/'scores.json',records);write(folder/'summary.json',summary)
    write(folder/'hwbscore.json', _hwbscore(records, summary, models, metrics))
    for model in models:
        scored_cases={r['case_id'] for r in records if r['model']==model and (r.get('score_0_1') is not None or r.get('features'))}
        if len(scored_cases)<BENCHMARK_CASES:
            print(f'WARNING: scores of model {model!r} cover {len(scored_cases)} of the {BENCHMARK_CASES} benchmark cases; '
                  'summary.json and hwbscore.json average only the scored cases and are not comparable with complete runs.',file=sys.stderr)
    return dict(output=str(folder.resolve()),records=len(records),models=models)
