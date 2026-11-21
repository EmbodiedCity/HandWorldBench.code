"""Durable task graph with explicit dependency failures and bounded retries."""
import json
import sqlite3
import time
from pathlib import Path

class Connection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


def connect(run):
    connection = sqlite3.connect(Path(run)/"queue.sqlite3",timeout=30,factory=Connection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL, state TEXT NOT NULL, ready REAL NOT NULL, attempt INTEGER NOT NULL DEFAULT 0, worker TEXT, result TEXT)")
    connection.execute("CREATE INDEX IF NOT EXISTS jobs_state_id ON jobs(state,id)")
    connection.commit()
    return connection

def enqueue(connection,tasks):
    for task in tasks:
        connection.execute("INSERT INTO jobs(id,payload,state,ready) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload WHERE jobs.state!='running'",(task['id'],json.dumps(task),"pending",time.time()))
    connection.commit()


def retry_failed(connection):
    """Open a bounded retry window while preserving previous attempt records."""
    retried=[]
    for row in connection.execute("SELECT * FROM jobs WHERE state IN ('failed','blocked')").fetchall():
        task=json.loads(row['payload'])
        task['retry_ceiling']=row['attempt']+3
        connection.execute("UPDATE jobs SET state='pending',payload=?,worker=NULL,ready=? WHERE id=?",
                           (json.dumps(task),time.time(),row['id']))
        retried.append(row['id'])
    connection.commit()
    return retried

def ready_jobs(connection,stage):
    states=dict(connection.execute("SELECT id,state FROM jobs"))
    responses=dict(connection.execute("SELECT id,result FROM jobs WHERE state IN ('failed','blocked')"))
    rows=list(connection.execute("SELECT * FROM jobs WHERE state='pending'"))
    result=[]
    for row in rows:
        if row['state']!='pending':continue
        task=json.loads(row['payload'])
        if stage=='observe' and task['kind']=='score':continue
        if stage=='score' and task['kind']!='score':continue
        dependency_states=[states.get(dep,'missing') for dep in task['dependencies']]
        if any(state in ('failed','blocked','missing') for state in dependency_states):
            binding=any(json.loads(responses.get(d) or '{}').get('status')=='input_binding_error' for d in task['dependencies'])
            value=dict(status='input_binding_error' if binding else 'dependency_failed',score_0_1=None,reason='Input binding error in required observation' if binding else 'Required observation failed',dependencies={d:states.get(d,'missing') for d in task['dependencies']})
            connection.execute("UPDATE jobs SET state='blocked',result=? WHERE id=?",(json.dumps(value),row['id']))
            states[row['id']]='blocked'
            responses[row['id']]=json.dumps(value)
        elif all(state=='done' for state in dependency_states):
            result.append((dict(row),task))
    connection.commit()
    return result

def priority(row,task,now,resident=()):
    aged=now-row['ready']>=1800
    return (0 if aged else 1,row['ready'] if aged else 0,-task.get('unlocks',0),-task.get('estimated_seconds',60),0 if task['backend'] in resident else 1,task['id'])


def remaining_paths(tasks, states, costs):
    """Estimate the longest unfinished downstream path from each task."""
    children = {task['id']: [] for task in tasks}
    by_id = {task['id']: task for task in tasks}
    for task in tasks:
        if states.get(task['id']) not in ('pending', 'running'):
            continue
        for dependency in task['dependencies']:
            children.setdefault(dependency, []).append(task['id'])
    memo = {}
    def visit(identifier):
        if identifier in memo:
            return memo[identifier]
        task = by_id[identifier]
        duration = costs.get(task['backend'], task.get('estimated_seconds', 60))
        memo[identifier] = duration + max((visit(child) for child in children[identifier]), default=0)
        return memo[identifier]
    return {identifier: visit(identifier) for identifier in by_id
            if states.get(identifier) in ('pending', 'running')}

