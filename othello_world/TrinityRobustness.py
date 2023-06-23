from maraboupy import MarabouCore, Marabou, MarabouUtils
from TrinityNet import TrinityNet
import torch
import pickle
import tempfile
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple
from settings import MARABOU_BIN
import os
import subprocess
import json
ZERO = 10**-6
NUM_THREADS = 30
TIMEOUT = 600

trinity = torch.load("trinity.pth", map_location=torch.device('cpu'))
with open("data_for_Marabou.pkl", "rb") as f:
    prepared_data = pickle.load(f)
print(prepared_data[0].keys())

def get_config():
    config_keys = [k for k,v in globals().items() if k.isupper()]
    # exec(open('configurator.py').read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    return config
# -----------------------------------------------------------------------------




def ge_constraint(x:int, y:int, prt:bool = False)->MarabouUtils.Equation:
    """
    return a MarabouUtils Equation of x>=y+ZERO
    """
    c = MarabouUtils.Equation(MarabouCore.Equation.GE)
    c.setScalar(ZERO)
    c.addAddend(1, x)
    c.addAddend(-1, y)
    if prt:
        print(c)
    return c

def trinity_robustness(input, sid:int, eps:float, sanity_check:bool = False):
    h = input["h"]
    true_label:int = input["argmax_logits"]
    other_labels = set(range(61))
    other_labels.remove(true_label)
    print(f"true label: {true_label}\nother labels: {other_labels}")
    print(f"running h through TrinityNet: {trinity(torch.Tensor(h))}")
    #convert to Marabou
    tempf = tempfile.NamedTemporaryFile(delete=False)
    torch.onnx.export(trinity, torch.Tensor(h), tempf.name, verbose=False)


    for other_l in other_labels:
        marabou_net = Marabou.read_onnx(tempf.name)

        ipq = marabou_net.getForwardQuery()

        input_vars: List[int] = np.array(marabou_net.inputVars).flatten().tolist()
        print(input_vars)
        output_vars: List[int] = np.array(marabou_net.outputVars).flatten().tolist()
        logit_vars: List[int] = output_vars[:61]
        probe_vars: List[int] = output_vars[61:]
        assert len(logit_vars) == 61
        assert len(probe_vars) == 192
        assert len(input_vars) == len(h)
        print("logit vars:", logit_vars)
        print("probe_vars:", probe_vars)
        logit_offset = logit_vars[0]
        probe_offset = probe_vars[0]

        print("setting bounds for input")
        for idx, v in enumerate(input_vars):
            ipq.setLowerBound(v, h[idx]-eps)
            ipq.setUpperBound(v, h[idx]+eps)


        print(input["reconstructed_board_logits"])
        print("adding constraints to make sure that the board stay the same")
        for i in range(probe_vars[0], probe_vars[-1], 3):
            print(i)
            cell_logits:List[float] = input["reconstructed_board_logits"][i-probe_offset:i+3-probe_offset]
            max_logits:float = max(cell_logits)

            if cell_logits[0]==max_logits:
                c1 = ge_constraint(i, i+1)
                c2 = ge_constraint(i, i+2) 

            elif cell_logits[1]==max_logits:
                c1 = ge_constraint(i+1, i)
                c2 = ge_constraint(i+1, i+2)

            elif cell_logits[2]==max_logits:
                c1 = ge_constraint(i+2, i)
                c2 = ge_constraint(i+2, i+1)

            ipq.addEquation(c1.toCoreEquation())
            ipq.addEquation(c2.toCoreEquation())
        # print("adding constraints for the network logits")
        # #sanity check: if we make the true argmax to be the max, Marabou should be able to basically recompute
        # #the true values for all nodes
        if sanity_check:
            c = ge_constraint(true_label+logit_offset, other_l + logit_offset)
            ipq.addEquation(c.toCoreEquation())
        else:
            #can other_label > true_label?
            c = ge_constraint(other_l + logit_offset, true_label + logit_offset)
            ipq.addEquation(c.toCoreEquation())


        ASSIGNMENT_FILE = f"{eps}/assignments_{sid}_{true_label}vs{other_l}.txt"
        queryFile = f"{eps}/finalQuery_{sid}_{true_label}vs{other_l}"

        MarabouCore.saveQuery(ipq, queryFile)

        #solve the query using Marabou binary
        marabouRes = subprocess.run([MARABOU_BIN, 
                                    f"--input-query={queryFile}",
                                    # "--snc",
                                    # f"--num-workers={NUM_THREADS}",
                                    "--export-assignment",
                                    f"--timeout={TIMEOUT}",
                                    ],
                                    capture_output=True,
                                    text=True, 
                                    timeout=TIMEOUT+10)
        #save results
        with open(f"{eps}/benchmark_config.json", "w") as f:
            json.dump(get_config(), f, indent=2)

        with open(f"{eps}/solving_stdout_{sid}_{true_label}vs{other_l}", "w") as f:
            f.write(marabouRes.stdout)

        with open(f"{eps}/solving_stderr_{sid}_{true_label}vs{other_l}", "w") as f:
            f.write(marabouRes.stderr)

        if os.path.exists("assignment.txt"):
            os.rename("assignment.txt", ASSIGNMENT_FILE)

EPS = 0.01
if not os.path.exists(f"{EPS}"):
    os.mkdir(f"{EPS}")
for sid, d in enumerate(prepared_data):
    trinity_robustness(d, sid, eps = EPS, sanity_check=False)