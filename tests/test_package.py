from pathlib import Path
import ast, hashlib, json, subprocess, unittest, tempfile, os, re
ROOT=Path(__file__).resolve().parents[1]
class PackageChecks(unittest.TestCase):
 def test_python_syntax(self):
  for folder in ['src','scripts','blog']:
   for f in (ROOT/folder).rglob('*.py'):ast.parse(f.read_text(),filename=str(f))
 def test_shell_syntax(self):
  for f in (ROOT/'run').glob('*.sh'):subprocess.run(['bash','-n',str(f)],check=True)
 def test_generation_slots(self):
  # Only env.sh owns the setting; training may temporarily force one slot.
  for f in (ROOT/'run').glob('*.sh'):
   if f.name != 'env.sh':
    self.assertFalse(any(line.startswith('SLOTS=') and ' pool ' not in line for line in f.read_text().splitlines()), f.name)
  for slots in (None, '1', '2'):
   env = os.environ.copy(); env.pop('SLOTS', None)
   if slots: env['SLOTS'] = slots
   result = subprocess.check_output(['bash', '-c', 'source "$1/run/env.sh"; echo "$SLOTS"', '_', str(ROOT)], env=env, text=True)
   self.assertEqual(result.strip(), slots or '1')
  with tempfile.TemporaryDirectory() as tmp:
   jobs = Path(tmp)/'jobs'
   jobs.write_text('echo "$GPU $MEM" > "'+tmp+'/assigned"\n')
   subprocess.run(['bash', '-c', 'source "$1/run/lib.sh"; GPUS=0 SLOTS=1 STAGGER=0 pool "$2"', '_', str(ROOT), str(jobs)], check=True, capture_output=True)
   self.assertEqual((Path(tmp)/'assigned').read_text().strip(), '0 0.85')
  script = (ROOT/'run/gen_qwen3.sh').read_text()
  expr = re.search(r'VLLM_GPU_MEMORY_UTILIZATION=(\S+)', script).group(1)
  for assignments, expected in [('unset QWEN3_MEM MEM;', '0.85'), ('unset QWEN3_MEM; MEM=0.425;', '0.425'), ('QWEN3_MEM=0.3; MEM=0.425;', '0.3')]:
   self.assertEqual(subprocess.check_output(['bash','-c', assignments+' echo '+expr], text=True).strip(), expected)
 def test_scored_record_hashes(self):
  for row in json.loads((ROOT/'results/astra_regime/scored/manifest.json').read_text()):
   b=(ROOT/row['path']).read_bytes()
   self.assertEqual(hashlib.sha256(b).hexdigest(),row['sha256'])
   self.assertEqual(len(b.splitlines()),row['rows'])
 def test_corrective_construction(self):
  # Exercise the original pure data-construction functions without loading torch/model weights.
  src=ast.parse((ROOT/'scripts/steer/loop_s_learn.py').read_text())
  names={'pair_texts','second_copy_char','doubt_text'}
  ns={'HEADER':"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n<think>\n{opening} "}
  exec(compile(ast.Module(body=[n for n in src.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'<construction>','exec'),ns)
  pairs=json.loads((ROOT/'scripts/steer/loop_short_pairs.json').read_text())['pairs']
  for pair in pairs:
   prod,rep=ns['pair_texts'](pair)
   start,_,copies=ns['second_copy_char'](prod,rep)
   for k in range(2,copies+1):
    text,boundary=ns['doubt_text'](prod,rep,k)
    self.assertEqual(text[boundary:],prod[start:])
    self.assertEqual(text[:boundary],prod[:start]+' '.join([pair['unit']]*k)+' ')
if __name__=='__main__':unittest.main()
