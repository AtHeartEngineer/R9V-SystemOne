import json,subprocess,sys,unittest
from pathlib import Path
class StreamTest(unittest.TestCase):
 def test_fragmented_stderr_does_not_swallow_stdout(self):
  child="import os,time; os.write(2,b'R9V_GPU {\"pid\":1,'); time.sleep(.05); os.write(1,b'{\"helper\":true}\\n'); time.sleep(.05); os.write(2,b'\"seq\":1}\\n')"
  # Feed literal newlines as escaped Python bytes, not a shell command.
  child=child.replace('\\\\n','\\n')
  result=subprocess.run([sys.executable,str(Path(__file__).with_name('line_stream.py')),sys.executable,'-c',child],capture_output=True,text=True,check=True,timeout=5)
  self.assertEqual(set(result.stdout.splitlines()), {'{"helper":true}','R9V_GPU {"pid":1,"seq":1}'})
if __name__=='__main__':unittest.main()
