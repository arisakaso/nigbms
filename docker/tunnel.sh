USER=$(whoami)
sudo chown $USER:$USER /home/$USER
pip install -e /home/$USER/nigbms
/workspace/code tunnel --no-sleep --accept-server-license-terms --name $TUNNEL_NAME