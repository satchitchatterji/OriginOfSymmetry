#!/usr/bin/env python3
import json
import subprocess
import pprint
import sys

output = subprocess.check_output(
    executable='jq', args=["-c"]+sys.argv[1:],
    shell=True, universal_newlines=True)
for row in output.split('\n'):
    try:
        j_row = json.loads(row)
        pprint.pprint(j_row)
    except:
        print(row)
