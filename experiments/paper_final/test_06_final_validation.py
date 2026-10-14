"""Delayed setup feedback must not change solve-RL costs or prefix state."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.paper_final import run_06_final_validation as validation
from experiments.paper_final.common.artifacts import digest, read
from experiments.paper_final.recovery.execution import EpisodeRecorder
from experiments.paper_final.recovery.joint_validation import copy_complete_prefix
from experiments.paper_final.recovery.retry_comparison import run_deferred_case
from experiments.paper_final.online.setup_branches import GenericBanditPolicy
from experiments.paper_final.online.case_loop import report_online_outcome
from experiments.paper_final.test_06_retry_screen import model_for, learn
from experiments.paper_final.test_06_recovery import outcome
from solve.tests.test_recursive_lstdq import RecursiveLstdqEpisodeTests


class JointValidationTests(unittest.TestCase):
    def test_setup_suffix_or_adjacent_never_changes_rl_attempt_cost(self):
        scenarios=[(['success'],None),(['nonconvergence','success'],None),
            (['nonconvergence','nonconvergence','success'],None),
            (['nonconvergence']*3,'success'),(['nonconvergence']*3,'nonconvergence')]
        for feedback in ('suffix_delayed','adjacent_delayed'):
            for statuses,fallback_status in scenarios:
                with self.subTest(feedback=feedback,statuses=statuses,fallback=fallback_status):
                    model=model_for('failure_model_suffix' if feedback=='suffix_delayed' else 'failure_model_adjacent')
                    controller=RecursiveLstdqEpisodeTests._controller()
                    expected=RecursiveLstdqEpisodeTests._controller()
                    recorder=EpisodeRecorder(controller)
                    seen=[];features=np.array([1.,.2,.4])
                    def solve(_params):
                        i=len(seen);seen.append(i)
                        recorder.start_episode(initial_environment_weight=1.)
                        recorder.update(features=features,action_index=0,cost=.02,
                            next_features=features,terminal=True)
                        recorder.finish_episode(learned=True)
                        return outcome(statuses[i],setup=0,solve=.1*(i+1))
                    def fallback():
                        self.assertIsNotNone(fallback_status)
                        return outcome(fallback_status,setup=0,solve=.4)
                    with patch('time.perf_counter',return_value=0.):
                        result=run_deferred_case(policy=GenericBanditPolicy(model),parameter_space={},
                            context=[1.,.5],solver=solve,fallback_solver=fallback,
                            variant=validation.screen.sweep.VARIANT,previous_update=0.,
                            before_attempt=lambda _:None,feedback_mode=feedback,recorder=recorder,snapshot_runtime=.003)
                    failed=fallback_status=='nonconvergence'
                    if not failed:
                        for i in range(len(statuses)):
                            expected.start_episode(initial_environment_weight=1.)
                            expected.update(features=features,action_index=0,cost=.02,
                                next_features=features,terminal=True)
                            expected.finish_episode(learned=True)
                    for key in ('a_matrix','a_inverse','b','theta','episode_moment_covariance'):
                        np.testing.assert_array_equal(getattr(controller,key),getattr(expected,key))
                    self.assertEqual(controller.steps,0 if failed else len(statuses))
                    self.assertEqual(model.t,len(statuses))
                    self.assertTrue(result.outcome['bandit_update_committed'])
                    self.assertEqual(result.outcome['controller_update_committed'],not failed)
                    self.assertEqual(result.outcome['replayed_transitions'],0)
                    self.assertEqual(result.outcome['target_replay_runtime'],0.)
                    self.assertTrue(all(a['rl_recovery_cost'] in (0.,None) for a in result.outcome['primary_attempts']))
                    native=sum(.1*(i+1) for i in range(len(statuses)))+(.4 if fallback_status else 0.)
                    reported=report_online_outcome(result.outcome,bandit_timing=result.timing)
                    self.assertAlmostEqual(reported['runtime'],native)
                    self.assertAlmostEqual(reported['end_to_end_runtime'],native+.003)
                    self.assertAlmostEqual(reported['case_learning_runtime'],.003)
                    validation.screen.sweep.audit_case(dict(problem_index=1,rl_active=False,
                        outcome={**reported,'primary_attempts':[{**a,'rl_episode_recorded':False} for a in reported['primary_attempts']]}),
                        None,feedback_mode=feedback)

    def test_complete_prefix_copy_keeps_last_arm_anchor_and_independent_state(self):
        source=model_for('failure_model_suffix')
        for context,cost in [([1.,.1],.8),([1.,.3],.4),([1.,.5],.7)]:
            source.predict(context);learn(source,cost,True)
        targets=[model_for('failure_model_suffix'),model_for('failure_model_suffix')]
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint=Path(temporary)/'prefix.npz';source.save_mutable_state(checkpoint)
            for target in targets:
                previous=copy_complete_prefix(source,target,checkpoint,previous_update=.0012)
                self.assertEqual(previous,.0012)
                self.assertEqual(digest([asdict(s) for s in source.history]),digest([asdict(s) for s in target.history]))
                np.testing.assert_array_equal(source._structured_anchor_arms(8),target._structured_anchor_arms(8))
                self.assertIsNot(source._cand,target._cand)
                self.assertIsNot(source.rng,target.rng)
            actions=[m.predict([1.,.7]) for m in (source,*targets)]
            self.assertEqual(actions[0],actions[1]);self.assertEqual(actions[1],actions[2])
            for key in ('A_inv','b','failure_b'):
                self.assertFalse(np.shares_memory(getattr(targets[0],key),getattr(targets[1],key)))
            learn(targets[0],.2,False)
            self.assertEqual(targets[0].t,targets[1].t+1)

    def test_preparation_has_both_controls_and_the_full_boundary_without_execution(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(validation.screen,'source_hashes',return_value={}), \
             patch.object(validation.screen.sweep,'native_hashes',return_value={}), \
             patch.object(validation.subprocess,'check_output',return_value='test'), \
             patch('hypre.bindings.create_env',side_effect=AssertionError('Preparation solved')):
            output=Path(temporary)/'study'
            validation.prepare(output,seeds=(5,6),feedback_mode='suffix_delayed')
            for seed in (5,6):
                stage=output/f'seed_{seed}';protocol=read(stage/'protocol.json')
                self.assertEqual((protocol['cases'],protocol['prefix_cases'],protocol['joint_cases']),(5000,1000,4000))
                self.assertEqual(protocol['methods'],list(validation.METHODS))
                self.assertFalse(protocol['rl_recovery_cost'])
                self.assertEqual(protocol['beta'],5.)
                self.assertEqual(len((stage/'inputs.jsonl').read_text().splitlines()),5000)
                for name in ('prefix','setup_only','default'):
                    config=read(stage/'configs'/f'{name}.json')
                    self.assertEqual(config['methods'][0]['solve'],'default')
                self.assertEqual(read(stage/'configs'/'setup_rl.json')['methods'][0]['solve_activation_case'],1000)

    def test_method_order_is_serial_and_balanced_within_each_phase(self):
        for i in range(1000):self.assertEqual(set(validation.order_for(i,1000)),{'prefix','default'})
        for block in range(1000,4995,6):
            orders=[validation.order_for(i,1000) for i in range(block,block+6)]
            self.assertEqual(len(set(orders)),6)
            for position in range(3):
                for method in validation.METHODS:
                    self.assertEqual(sum(row[position]==method for row in orders),2)

    def test_rl_counter_audit_excludes_rolled_back_cases_and_unrecorded_attempts(self):
        def case(committed,attempts):
            return dict(outcome=dict(controller_update_committed=committed,primary_attempts=attempts))
        observed=dict(rl_episode_recorded=True,cycle_actions=[1.,1.2,1.])
        unobserved=dict(rl_episode_recorded=False,cycle_actions=[])
        rows=[case(True,[observed,unobserved,observed]),case(False,[observed])]
        validation.audit_controller_counts(rows,dict(steps=6,episodes=2))
        for summary in (dict(steps=9,episodes=3),dict(steps=6,episodes=3),dict(steps=5,episodes=2)):
            with self.assertRaises(ValueError):validation.audit_controller_counts(rows,summary)


if __name__=='__main__':unittest.main()
