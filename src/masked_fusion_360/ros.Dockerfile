FROM mrt_pytorch_ros_base:latest

RUN pip install vit-pytorch==1.5.3 pytorch_lightning==2.0.0 "typer[all]"

ADD . .

CMD /bin/bash -c "source /opt/ros/noetic/setup.bash && python3 ros_node.py"
