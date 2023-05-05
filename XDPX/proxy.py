import sys
from datetime import datetime
from subprocess import Popen, PIPE, STDOUT


def cli_main():
    args = ' '.join(sys.argv[1:])

    status = None
    try:
        with Popen(['odpscmd', '-e', args], stdout=PIPE, stderr=STDOUT, universal_newlines=True) as p:
            for line in p.stdout:
                line = line.rstrip()
                if line.startswith('pytorch') or line.startswith('train_aon'):
                    new_status = line[30:]
                    if new_status == status:
                        print(line, end='\r')
                    else:
                        print(line)
                        status = new_status
                else:
                    if status:
                        print()
                        status = None
                    print(line)
                    if line.startswith('http'):
                        with open('command_history.txt', 'a') as f:
                            f.write(args + '\n')
                            f.write(str(datetime.now()).split('.')[0] + '\n')
                            f.write(line + '\n\n')
                            f.flush()
    except KeyboardInterrupt:
        return


if __name__ == '__main__':
    cli_main()
