import pandas as pd
import re


def main():
    df = pd.read_csv('data/raw_table.csv')
    i, ii, iii, iv = [], [], [], []
    master = [i, ii, iii, iv]
    pattern = re.compile('EPI_ISL_')
    # Filter GISAID ids
    for k in range(len(df)):
        accession = df.iloc[k]['accession']
        y = df.iloc[k]['lineage'][0]
        if not re.match(pattern, accession):
            if y == '1':
                i.append(accession)
            elif y == '2':
                ii.append(accession)
            elif y == '3':
                iii.append(accession)
            elif y == '4':
                iv.append(accession)
            else:
                continue

    # Write lists to text files
    j = 1
    for ele in master:
        with open(f'accession{j}.txt', 'w') as file:
            for item in ele:
                file.write(f"{item}\n")
        j += 1

    print('Done')


if __name__ == "__main__":
    main()
