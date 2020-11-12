import os 

def replace_version(old_version, new_version):
    if not isinstance(old_version, tuple) or not isinstance(new_version, tuple):
        raise ValueError('`old_version` and `new_version` must be a version tuple. Eg: (1.2.3)')

    major, minor, micro = old_version[:3]
    old_version = f'{major}.{minor}.{micro}'
    major, minor, micro = new_version[:3]
    new_version = f'{major}.{minor}.{micro}'
    print('New version', new_version)

    for root, _, files in os.walk('../caer'):
        for file in files:
            if file.endswith(('.py', '.cpp', '.c', '.h', '.hpp')):
                with open(os.path.abspath(os.path.join(root, file)), 'r') as f:
                    new_text = f.read().replace('version ' + old_version, 'version ' + new_version)

                with open(os.path.abspath(os.path.join(root, file)), 'w') as f:
                    print(os.path.abspath(os.path.join(root, file)))
                    f.write(new_text)

replace_version((1,8,0), (3,9,1))

