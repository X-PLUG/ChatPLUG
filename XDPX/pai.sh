set -e

n_gpus=""
n_workers=""

ARGS=""
if [[ -f .pai_config ]]; then
  rm .pai_config
fi
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -v|--pt_version)
        pt_version="$2"
        echo "pt_version=$2" >> .pai_config
        shift 2
        ;;
        -g|--n_gpus)
        n_gpus="$2"
        echo "n_gpus=$2" >> .pai_config
        shift 2
        ;;
        -w|--n_workers)
        n_workers="$2"
        echo "n_workers=$2" >> .pai_config
        shift 2
        ;;
        -c|--n_cpus)
        n_cpus="$2"
        echo "n_cpus=$2" >> .pai_config
        shift 2
        ;;
        -m|--memory)
        memory="$2"
        echo "memory=$2" >> .pai_config
        shift 2
        ;;
        -s|--v100)
        v100="$2"
        echo "v100=$2" >> .pai_config
        shift 2
        ;;
        -t|--tables)
        tables="$2"
        shift 2
        ;;
        -r|--test_root)
        test_root="$2"
        shift 2
        ;;
        *)
        ARGS=$ARGS" "$key
        shift
        ;;
    esac
done

ARGS=$(echo "$ARGS" | xargs)

if [[ -f .test_meta  ]]; then
  prev_meta=$(<.test_meta)
else
  prev_meta="/tmp/outputs/debug/"
fi
entry='xdpx/run.py'
if [[ $ARGS == 'coverage' ]] ; then
  echo 'ERROR: report save path needed for coverage test'
  exit 1
fi
if [[ $ARGS == test* ||  $ARGS == benchmark* || $ARGS == coverage* ]] ; then
  entry='tests/test_legacy.py'
  if [ -z "$test_root" ] ; then
    while true; do
      read -rp "Specify oss test root: " test_root
      if [[ $test_root == oss://* && $(x-io is_writable "$test_root") == True ]] ; then
        break
      else
        echo "\"$test_root\" cannot be created on OSS."
      fi
    done
  else
    if [[ $test_root == oss://* && $(x-io is_writable "$test_root") == False ]] ; then
      echo "\"$test_root\" cannot be created on OSS."
      exit
    fi
  fi
  echo "$test_root" > .test_meta
else
  IFS=' ' read -ra cmds <<< "$(printf '%s\n' "$ARGS" | grep -ve '--' | tr '\n' ' ')"
  # shellcheck disable=SC2086
  python $entry ${cmds[*]} --dry
fi

if [ -d .git ]; then
    git log -n1 --format=format:"%H" > .git_version
    echo "" >> .git_version
    git status >> .git_version
fi

ZIP_FILE="../xdpx.tar"
if [ -f ZIP_FILE ]; then
   rm $ZIP_FILE;
fi

tar -cf $ZIP_FILE --exclude=./.* *;
tar -rf $ZIP_FILE .git_version
tar -rf $ZIP_FILE .coveragerc
if [ -e .test_meta ]; then
  tar -rf $ZIP_FILE .test_meta
fi
gzip -f $ZIP_FILE
ZIP_FILE=$ZIP_FILE.gz
echo "$prev_meta" > .test_meta

if (( $( wc -c "${ZIP_FILE}" | awk '{print $1}' ) > 100*1024*1024 )); then
  echo "ERROR: \"$(pwd)\" has exceeded 100M. Please clean the XDPX directory to enable fast & stable PAI submission."
  exit
fi

python pai_config.py
. .pai_config

if [[ $n_workers -gt 1 && $pt_version == '131' ]]; then
  docker_fusion=true
  n_workers=$((n_workers * n_gpus))
  n_gpus=1
else
  docker_fusion=false
fi

# https://yuque.antfin-inc.com/pai-user/manual/pytorch_get_started
# n_cpus and memory will be 18 / 60 if not specified
# -project algo_public_dev
PAI_COMMAND="pai -name pytorch${pt_version}
    -Dscript='file:///$PWD/$ZIP_FILE'
    -DentryFile=$entry
    -DuserDefinedParameters='$ARGS'
    -Dpython='3.6'
    -Dcluster='{\"worker\":{\"cpu\":${n_cpus}00, \"memory\":${memory}, \"gpu\":${n_gpus}00}}'
    -DworkerCount=$n_workers
    "
if [[ $docker_fusion == 'true' ]]; then
  PAI_COMMAND=$PAI_COMMAND$'-DenableDockerFusion=true\n'
fi

if [[ $v100 == 'y' ]] ; then
  PAI_COMMAND=$'set odps.algo.hybrid.deploy.info=LABEL:V100:OPER_EQUAL;\n'$PAI_COMMAND
fi

if [[ -n "$tables" ]] ; then
  PAI_COMMAND=$PAI_COMMAND$"-Dtables='$tables'"$'\n'
fi

PAI_COMMAND=$PAI_COMMAND';'

echo 
echo "$PAI_COMMAND"

python proxy.py "$PAI_COMMAND"

echo 

if [ -f $ZIP_FILE ]; then
   rm $ZIP_FILE;
fi
