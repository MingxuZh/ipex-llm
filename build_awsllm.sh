# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
            #  --build-arg TORCHCCL_WHEEL=oneccl_bind_pt-*.whl  \
#

set -e
set -x

docker_name=forllmaws

echo "docker_name: ${docker_name}"
IMAGE_NAME=${docker_name}
ipex_branch=main



sudo docker build --no-cache \
             -t $IMAGE_NAME \
             -f DockerFileaws.llm .
