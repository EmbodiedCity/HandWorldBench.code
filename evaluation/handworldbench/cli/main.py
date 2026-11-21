"""Evaluate a model, recover a run, or compare completed runs."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','observe','score','evaluate','resume','status','report'])
    p.add_argument('--dataset');p.add_argument('--model',help='Name of the evaluated model (defaults to "reference" with --use-reference-videos)');p.add_argument('--config')
    source=p.add_mutually_exclusive_group()
    source.add_argument('--videos');source.add_argument('--predictions');source.add_argument('--use-reference-videos',action='store_true')
    p.add_argument('--cases');p.add_argument('--metrics');p.add_argument('--run-dir');p.add_argument('--gpus',default='auto')
    p.add_argument('--foreground',action='store_true');p.add_argument('--runs',nargs='+');p.add_argument('--output')
    args=p.parse_args(argv)
    try:
        from handworldbench.registry import select
        if args.metrics:select(args.metrics)
        if args.command=='report':
            from handworldbench.reporting.report import report
            runs=args.runs or ([args.run_dir] if args.run_dir else [])
            if not runs:p.error('report requires --runs or --run-dir')
            from handworldbench.runtime.storage import read
            print(json.dumps(report(runs,args.output,read(args.cases) if args.cases else None,select(args.metrics) if args.metrics else None),ensure_ascii=False));return
        if not args.run_dir:p.error('--run-dir is required')
        inputs=bool(args.videos or args.predictions or args.use_reference_videos)
        if args.use_reference_videos and not args.model:args.model='reference'
        if args.command in ('prepare','evaluate') and (inputs or args.command=='prepare'):
            if not args.dataset or not args.model or not inputs:p.error('New runs require --dataset, --model, and an input mode')
            from handworldbench.runtime.prepare import prepare
            print(json.dumps(prepare(args),ensure_ascii=False))
            if args.command=='prepare':return
        elif any([args.dataset,args.model,args.config,args.metrics,args.cases,inputs]):p.error('Input changes require prepare or evaluate with a complete input set')
        run=Path(args.run_dir).resolve()
        if not (run/'configs/run.json').exists():p.error('Run is not prepared')
        from handworldbench.runtime.storage import read
        if read(run/'configs/run.json').get('imported_results') and args.command not in ('status','report'):
            p.error('This directory contains collected results; resume the source evaluation run')
        if args.command=='status':
            from handworldbench.runtime.queue import connect
            from handworldbench.runtime.scheduler import active
            with connect(run) as c:counts=dict(c.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
            print(json.dumps(dict(run_dir=str(run),scheduler=active(run),tasks=counts)));return
        from handworldbench.runtime.scheduler import launch
        print(json.dumps(launch(run,args.command if args.command in ('observe','score') else 'evaluate',args.foreground,retry=args.command=='resume')))
    except (ValueError,FileNotFoundError) as exc:p.error(str(exc))

