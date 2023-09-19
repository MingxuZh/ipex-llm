
currentBuild.displayName = NODE_LABEL

node(NODE_LABEL) {
    deleteDir()
    sh '''
        #!/bin/bash
        export http_proxy=http://localhost:911
        export https_proxy=http://localhost:912
        git config --global http.proxy http://localhost:911
        git config --global https.proxy http://localhost:912
        set -xe
        if [ $(echo ${M_TOOL} |grep -w "mlc" |wc -l) -ne 0 ];then
            wget http://mengfeil-ubuntu.sh.intel.com/share/mlc_v3.9a_internal.tgz
            tar xf mlc_v3.9a_internal.tgz
            cd Linux
            sudo ./mlc_internal --max_bandwidth > mlc.log
            grep "Stream-triad like:" mlc.log
            bw=$(grep "Stream-triad like:" *.log |awk 'BEGIN{sum=0;num=0}{sum+=$3;num++;}END{print sum/num;}')
        elif [ $(echo ${M_TOOL} |grep -w "stream" |wc -l) -ne 0 ];then
            wget http://mengfeil-ubuntu.sh.intel.com/share/stream.tgz
            tar xf stream.tgz > /dev/null 2>&1
            cd stream
            gcc -fopenmp -D_OPENMP stream.c -o stream
            cores_per_socket=$(lscpu |grep 'Core(s) per socket:' |sed 's/[^0-9]//g')
            export OMP_NUM_THREADS=${cores_per_socket}
            for i in $(seq 10)
            do
                ./stream > $i.log
            done
            grep "Triad:" *.log
            bw=$(grep "Triad:" *.log |awk 'BEGIN{sum=0;num=0}{sum+=$2;num++;}END{print sum/num;}')
        else
            echo "No such tool: ${M_TOOL}, please use mlc or stream"
            exit 1
        fi

        echo "${NODE_LABEL}: ${bw} MB/s" > ${WORKSPACE}/result.txt
    '''
    currentBuild.description = readFile file: 'result.txt'
}