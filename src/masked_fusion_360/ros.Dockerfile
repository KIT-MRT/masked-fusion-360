FROM mrt_pytorch_ros_base:latest

ADD . .

CMD source /opt/ros/noetic/setup.bash && python3 ros_node.py