import os

link_file = 'ori_check_links.txt'

bad_links_file = 'check_links.txt'

bad_links = []

with open(link_file, 'r') as f:
    for line in f:
        url = line.strip()
        if not url:
            continue

        filename = url.split('?')[0].split('/')[-1]

        if not os.path.exists(filename) or os.path.getsize(filename) < 10:
            bad_links.append(url)

print(f"{len(bad_links)} files near 0 MB")

with open(bad_links_file, 'w') as f:
    for url in bad_links:
        f.write(url + '\n')

print(f"Results in {bad_links_file}")

