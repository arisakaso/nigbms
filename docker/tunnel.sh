USER=$(whoami)
sudo chown $USER:$USER /home/$USER
/workspace/code tunnel --no-sleep --accept-server-license-terms --name $TUNNEL_NAME