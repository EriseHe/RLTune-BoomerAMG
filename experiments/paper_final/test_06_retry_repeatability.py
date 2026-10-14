"""Check timing order, exact reference values and non-invasive diagnostics."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.paper_final import run_06_retry_repeatability as serial
from experiments.paper_final.common.artifacts import dump, read
from experiments.paper_final.test_06_retry_screen import model_for, learn
from setup.space import build_setup_parameter_spec


class RepeatabilityTests(unittest.TestCase):
    def test_every_input_has_each_method_in_each_position_across_repeats(self):
        for index in range(1000):
            orders = [serial.method_order(index, repetition) for repetition in range(3)]
            for order in orders:self.assertEqual(set(order),set(serial.METHODS))
            for position in range(3):
                self.assertEqual({order[position] for order in orders},set(serial.METHODS))
        self.assertEqual(len({serial.method_order(i,0) for i in range(6)}),6)

    def test_diagnostics_preserve_choices_updates_and_random_state(self):
        for rule in (n for n in serial.ALL_METHODS if n != "default"):
            base=model_for(rule)
            observed=serial.observed_model(model_for(rule))
            for case in range(15):
                context=[1,(case+1)/16]
                actions=[m.predict(context) for m in (base,observed)]
                self.assertEqual(actions[0],actions[1])
                arms=[m._last_arm for m in (base,observed)]
                observations=[m.defer_cost(False) for m in (base,observed)]
                retry=[m.select_recovery(context,excluded_arms=[a]) for m,a in zip((base,observed),arms)]
                self.assertEqual(retry[0],retry[1])
                second=[m.defer_cost(True) for m in (base,observed)]
                for m,first,last in zip((base,observed),observations,second):
                    m.resolve_cost(first,.4+case*.01);m.resolve_cost(last,.2+case*.01)
                keys = ("A_inv","b","failure_b") if hasattr(base,"failure_b") else ("A_inv","b")
                for key in keys:
                    np.testing.assert_array_equal(getattr(base,key),getattr(observed,key))
                self.assertEqual(base.rng.bit_generator.state,observed.rng.bit_generator.state)
            self.assertEqual(len(observed.selection_diagnostics),30)
            for record in observed.selection_diagnostics:
                if record["strategy"]!="initial_guess":
                    self.assertIn(record["selected"]["arm"],record["candidate_arms"])
                    self.assertGreaterEqual(record["best_two_gap"],-1e-12)

    def test_single_run_order_matches_module04_rng_without_learner_rng(self):
        for seed in (56772120,173397802,663558482):
            protocol=dict(cases=1000,methods_order=serial.ALL_METHODS,
                          order_rule=serial.MODULE04_ORDER,seeds=dict(method_order=seed))
            expected_rng=np.random.default_rng(seed)
            expected=[tuple(serial.ALL_METHODS[int(i)] for i in expected_rng.permutation(5))
                      for _ in range(1000)]
            self.assertEqual(serial.execution_orders(protocol),expected)
            self.assertEqual(serial.execution_orders(protocol),expected)
            self.assertTrue(all(set(order)==set(serial.ALL_METHODS) for order in expected))

    def test_one_run_all_four_rules_uses_original_seed_bundle_without_solves(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(serial.screen,"source_hashes",return_value={}), \
             patch.object(serial.screen.sweep,"native_hashes",return_value={}), \
             patch.object(serial.screen.subprocess,"check_output",return_value="test"), \
             patch("hypre.bindings.create_env",side_effect=AssertionError("Unexpected solve")):
            output=Path(temporary)/"study"
            serial.prepare(output,cases=10,seeds=(1,2,3),repetitions=1,
                           methods=serial.ALL_METHODS,module04_order=True,concurrent_seeds=True)
            plan=read(output/"sequence_protocol.json")
            self.assertEqual([s['name'] for s in plan['stages']],['seed_1','seed_2','seed_3'])
            self.assertEqual(plan['repetitions'],1)
            self.assertTrue(plan['concurrent_seeds'])
            for stage in plan['stages']:
                protocol=read(output/stage['name']/"protocol.json")
                original=read(serial.screen.SEED_CONFIGS[stage['seed']])
                self.assertEqual(protocol['seeds'],original['seeds'])
                self.assertEqual(protocol['seed_groups'],original['stream']['seed_groups'])
                self.assertEqual(protocol['shuffle_seeds'],original['stream']['shuffle_seeds'])
                self.assertEqual(set(protocol['branches']),set(serial.ALL_METHODS))
                self.assertEqual(protocol['total_attempt_budget'],4)
                self.assertEqual(protocol['order_rule'],serial.MODULE04_ORDER)
                self.assertEqual(protocol['rl_steps_expected'],0)
                self.assertEqual(len(set(c['inputs_sha256'] for c in protocol['preparation_checks'].values())),1)

    def test_concurrent_dispatch_starts_every_seed_before_polling(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            stages=[]
            for seed in (1,2,3):
                name=f'seed_{seed}'
                dump(output/name/'protocol.json',dict(seed=seed))
                dump(output/name/'status.json',dict(status='complete'))
                stages.append(dict(name=name,protocol_sha256=serial.file_hash(output/name/'protocol.json')))
            launched=[]
            class Worker:
                def __init__(self,pid):self.pid=pid
                def poll(self):
                    self_check.assertEqual(len(launched),3)
                    return 0
            self_check=self
            def launch(*args,**kwargs):
                worker=Worker(len(launched)+1)
                launched.append(worker)
                return worker
            with patch.object(serial.subprocess,'Popen',side_effect=launch):
                serial.run_concurrent_stages(output,dict(stages=stages))
            self.assertEqual(len(launched),3)
            self.assertEqual(len(read(output/'active_workers.json')),3)

    def test_five_method_serial_audit_checks_module04_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            protocol=dict(cases=2,repetition=0,methods_order=serial.ALL_METHODS,
                          order_rule=serial.MODULE04_ORDER,seeds=dict(method_order=123))
            dump(output/'protocol.json',protocol)
            rows=[dict(input_index=index,method=name,pid=1,start_ns=10*(index*5+position),
                       end_ns=10*(index*5+position)+9)
                  for index,order in enumerate(serial.execution_orders(protocol))
                  for position,name in enumerate(order)]
            (output/'execution.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            serial.audit_execution(output)
            audit=read(output/'execution_audit.json')
            self.assertEqual(audit['method_cases'],10)
            self.assertEqual(audit['overlap_count'],0)
            self.assertTrue(audit['module04_method_order'])
            rows[1]['start_ns']=8
            (output/'execution.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'Overlapping'):
                serial.audit_execution(output)

    def test_exact_reference_keeps_grid_positions_and_other_values(self):
        spec,_=build_setup_parameter_spec(tune_dim=7,parameter_resolution=20)
        parameters={p.name:p for p in spec.parameters}
        old=np.linspace(0,.95,20)
        expected_threshold=old.copy();expected_threshold[5]=.25
        expected_rows=old.copy();expected_rows[0]=1e-6;expected_rows[18]=.9
        np.testing.assert_array_equal(parameters['strong_threshold'].values,expected_threshold)
        np.testing.assert_array_equal(parameters['max_row_sum'].values,expected_rows)
        self.assertEqual(len(parameters['strong_threshold'].values),20)
        self.assertEqual(len(parameters['max_row_sum'].values),20)

    def test_prepare_contains_only_two_paired_feedback_rules_and_default(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(serial.screen,"source_hashes",return_value={}), \
             patch.object(serial.screen.sweep,"native_hashes",return_value={}), \
             patch.object(serial.screen.subprocess,"check_output",return_value="test"), \
             patch("hypre.bindings.create_env",side_effect=AssertionError("Unexpected solve")):
            output=Path(temporary)/"study"
            serial.prepare(output,cases=10,seeds=(5,))
            plan=read(output/"sequence_protocol.json")
            self.assertEqual(len(plan['stages']),3)
            for stage in plan['stages']:
                protocol=read(output/stage['name']/"protocol.json")
                self.assertEqual(set(protocol['branches']),set(serial.METHODS))
                self.assertEqual(protocol['feedback'],dict(failure_model_suffix='suffix_delayed',failure_model_adjacent='adjacent_delayed'))
                self.assertEqual(protocol['total_attempt_budget'],4)
                self.assertEqual(protocol['execution'],'serial_complete_cases')
                self.assertEqual(protocol['rl_steps_expected'],0)

    def test_serial_audit_rejects_overlap_and_wrong_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            dump(output/'protocol.json',dict(cases=1,repetition=0))
            rows=[dict(input_index=0,method=name,pid=1,start_ns=10*i,end_ns=10*i+9)
                  for i,name in enumerate(serial.method_order(0,0))]
            def write():
                (output/'execution.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            write();serial.audit_execution(output)
            rows[1]['start_ns']=8;write()
            with self.assertRaisesRegex(ValueError,'Overlapping'):serial.audit_execution(output)
            rows[1]['start_ns']=10
            rows[0]['method'],rows[1]['method']=rows[1]['method'],rows[0]['method'];write()
            with self.assertRaisesRegex(ValueError,'order'):serial.audit_execution(output)


if __name__=='__main__':unittest.main()
