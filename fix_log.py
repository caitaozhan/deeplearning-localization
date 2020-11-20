from input_output import IOUtility, Input, Output
from server import Server

logs = ['result/11.19/log-differentsensor']
output_dir = 'result/11.19'
output_file = 'log-differentsensor2'
server = Server(output_dir, output_file)

f = open(logs[0], 'r')

while True:
    line = f.readline()
    if line == '':  # EOF
        break
    myinput = Input.from_json_str(line)
    outputs = []
    i = 1
    line = f.readline()
    while line != '' and line != '\n':
        output = Output.from_json_str(line)
        if i == 2:
            output.method = 'dl2'
        i += 1
        outputs.append(output)
        line = f.readline()
    server.log(myinput, outputs)
