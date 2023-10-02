FROM mrt_pytorch_ros_base:latest

RUN pip install vit-pytorch pytorch_lightning "typer[all]"

ADD . .

CMD /bin/bash -c "source /opt/ros/noetic/setup.bash && python3 ros_node.py"