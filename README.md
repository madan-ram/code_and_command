#files information#

placeing_center.py -> containe code to center image inside fixed frame size.

caffe_feature_extract.py -> explain how to fedforward neural network to certain layer and extract feature for searching,I also display the image feature visualization usingt-sne (2d embbeding)  etc ...

data_augmentation.py -> containe code to  augment data by transformation such as {rotation, scaling}.

feature_heat_map -> A directory which has code to generate the heat map for high level feature of the image; which is effecting prediction result. { run$ python main.py <image_path>}

How make ubuntu PC act as internet gateway

	Enter following command to edit interfaces file:
	sudo vim /etc/network/interfaces

	Edit the file with the following lines: (add your netmask and gateway)
	auto lo 
	iface lo inet loopback

	auto eth0
	iface eth0 inet static
	address 182.x.x.x 
	netmask  x.x.x.x 
	gateway x.x.x.x

	Restart you network manager
	sudo service network-manager restart

	Now edit /etc/sysctl.conf and uncomment:

	# net.ipv4.ip_forward=1
	so that it reads:
	net.ipv4.ip_forward=1

	You want to run "sudo sysctl -p /etc/sysctl.conf" to make the new value take effect.