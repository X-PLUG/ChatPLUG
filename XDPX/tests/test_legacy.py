import sys
import traceback
from xdpx.utils import io, compress_dir

"""
run pipeline tests where cli commands are not available (like on PAI).
"""


def cli_main(argv=sys.argv):
    if argv[1] == 'coverage':
        if len(argv) == 2:
            print('Usage: coverage <path to report dir>')
            exit()
        export_path = argv[2]
        if not export_path.endswith('.zip'):
            export_path += '.zip'
        import coverage
        cov = coverage.Coverage(config_file=True)
        cov.erase()
        cov.start()
        cli_main(['', 'test'])
        cov.stop()
        cov.save()
        cov.combine()
        cov.html_report()
        compress_dir('user/htmlcov')
        io.copy('user/htmlcov.zip', export_path)
        return

    # import after coverage start to count the real coverage
    from xdpx.run import main
    commands = []

    if argv[1] == 'test':
        test_script = 'tests/test_pipeline.sh'
        if '--tensorflow' not in argv:
            argv.append('--tensorflow')
    elif argv[1] == 'benchmark':
        test_script = 'tests/test_benchmark.sh'
    else:
        raise ValueError(f'Undefined command: {argv[1]}')
    with open(test_script) as f:
        for line in f:
            if line.startswith('x-'):
                commands.append(line.rstrip())
    # simulate cli call
    with open('.test_meta', 'r') as f:
        outdir = f.read().strip()
    if io.exists(outdir) and argv[1] == 'test':
        io.rmtree(outdir)
    for command in commands:
        try:
            main(['xdpx/run.py'] + command.split() + argv[2:])
        except Exception:
            print(f'Test "{command}" failed.')
            print(traceback.format_exc())
    print('\nTests complete. Search for "Error" to check runtime errors in tests.')


if __name__ == "__main__":
    cli_main()
