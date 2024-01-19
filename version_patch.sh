#!/bin/bash

cd frameworks.ai.pytorch.ipex-cpu
#LN=$(grep "torch-ccl:" -n dependency_version.yml | cut -d ":" -f 1)
#LN=$((${LN}+3))
#sed -i "${LN}s/.*/  commit: ccl_torch_2.2.0_rc1/" dependency_version.yml
sed -i "s/URL_NIGHTLY=\"\"/URL_NIGHTLY=\"test\/\"/" scripts/compile_bundle.sh
sed -i "s/URL_NIGHTLY=\"\"/URL_NIGHTLY=\"test\/\"/" docker/Dockerfile.compile
sed -i "s/URL_NIGHTLY=\"\"/URL_NIGHTLY=\"test\/\"/" examples/cpu/inference/python/llm/tools/env_setup.sh
sed -i "99s/.*/.../" examples/cpu/inference/python/llm/tools/env_setup.sh
sed -i "100s/.*/.../" examples/cpu/inference/python/llm/tools/env_setup.sh
sed -i "101s/.*/.../" examples/cpu/inference/python/llm/tools/env_setup.sh
sed -i "102s/.*/.../" examples/cpu/inference/python/llm/tools/env_setup.sh
