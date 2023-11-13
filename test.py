import tarfile
file = tarfile.open("BuycraftX.tar.gz", mode='r')
total_members = len(file.getmembers())
for member in file.getmembers():
    file.extract(member, 'out_bc')
