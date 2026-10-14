"""Execution and verification helpers for the official studies."""
from __future__ import annotations
def audit_controller_counts(rows, summary):
    retained=[attempt for row in rows if row['outcome']['controller_update_committed']
              for attempt in row['outcome']['primary_attempts'] if attempt.get('rl_episode_recorded',False)]
    steps=sum(len(attempt['cycle_actions']) for attempt in retained)
    if summary['steps']!=steps or summary['episodes']!=len(retained):
        raise ValueError('RL counters differ from retained attempt observations')
