import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from handworldbench.runtime.queue import connect,enqueue
from handworldbench.runtime.scheduler import run_queue
from handworldbench.runtime.storage import write,read


class CompletedReportTests(unittest.TestCase):
    def test_evaluate_exports_scores_including_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            for folder in ('configs','logs','reports','results','workers'):(run/folder).mkdir()
            write(run/'configs/run.json',dict(runtime={'cache_root':str(run/'cache')},devices=[]))
            value=dict(case_id='case',model='Model',category='human',metric_id='musiq',metric_protocol='1',raw_score=0.,score_0_1=0.,status='ok')
            write(run/'configs/slots.json',[dict(value,task_id='score')])
            task=dict(id='score',kind='score',backend='score',metric_id='musiq',dependencies=[],estimated_seconds=1)
            with connect(run) as connection:
                enqueue(connection,[task])
                connection.execute("update jobs set state='done',result=?",(json.dumps(dict(status='ok',value=value)),))
            handlers={key:signal.getsignal(key) for key in (signal.SIGTERM,signal.SIGINT)}
            try:
                with patch('handworldbench.runtime.scheduler.subprocess.check_output',return_value=''):
                    run_queue(run,'evaluate')
            finally:
                for key,handler in handlers.items():signal.signal(key,handler)
            self.assertEqual(read(run/'reports/scores.json')[0]['score_0_1'],0.)
            self.assertEqual(read(run/'reports/summary.json')[0]['common_cases'],1)


if __name__=='__main__':unittest.main()
