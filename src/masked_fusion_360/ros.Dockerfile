FROM mrt_pytorch_ros_base:latest

RUN pip install vit-pytorch pytorch_lightning

ADD . .

CMD source /opt/ros/noetic/setup.bash && python3 ros_node.py