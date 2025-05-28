import csv
import time
import os
from Bio import Entrez, SeqIO


def download_fasta(accession_number, email, output_directory):
    Entrez.email = email

    # Create output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_file = f"{output_directory}/{accession_number}.fasta"

    # Check if the file already exists
    if os.path.exists(output_file):
        print(f"FASTA sequence '{output_file}' already exists. Skipping download.")
        return

    try:
        handle = Entrez.efetch(db="nucleotide", id=accession_number, rettype="fasta", retmode="text")
        record = handle.read()
        handle.close()

        with open(output_file, "w") as file:
            file.write(record)

        print(f"FASTA sequence saved successfully as '{output_file}'")
    except Exception as e:
        print(f"Error fetching sequence for {accession_number}: {str(e)}")


def process_csv(file_path, email, output_directories):
    count = 0

    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip the header

    for row in reader:
        for j, accession_number in enumerate(row):
            accession_number = accession_number.strip()  # Remove any whitespace
            if accession_number:
                download_fasta(accession_number, email, output_directories[j])
                count += 1

                # Introduce a 1-second delay after every 3 sequences
                # To abide by the rules of downloading from ncbi
                if count % 3 == 0:
                    time.sleep(1)


def process_text(file_path, email, output_directories):
    count = 0

    with open(file_path, 'r') as textfile:
        content = textfile.readlines()

        for line in content:
            accession_number = line.strip()

            if accession_number:
                download_fasta(accession_number, email, output_directories)
                count += 1

                # Introduce a 1-second delay after every 3 sequences
                # To abide by the rules of downloading from ncbi
                if count % 3 == 0:
                    time.sleep(1)


def download_ncbi_fasta(query, num_results, output_dir, email):
    """
    Searches NCBI for a given query, retrieves the specified number of results,
    and downloads them as FASTA files, saving each with its accession ID.

    Parameters:
        query (str): Search term for NCBI (e.g., "Homo sapiens COX1")
        num_results (int): Number of sequences to retrieve
        output_dir (str): Directory to save FASTA files (default: "NCBI_FASTA")
    """
    # Providing email id
    Entrez.email = email

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Searching NCBI for '{query}'...")

    # Search for sequence IDs
    search_handle = Entrez.esearch(db="nucleotide", term=query, retmax=num_results)
    search_results = Entrez.read(search_handle)
    search_handle.close()

    accession_ids = search_results["IdList"]

    if not accession_ids:
        print("No results found!")
        return

    print(f"Found {len(accession_ids)} results. Downloading FASTA files...")

    # Fetch sequences
    fetch_handle = Entrez.efetch(db="nucleotide", id=accession_ids, rettype="fasta", retmode="text")
    records = SeqIO.parse(fetch_handle, "fasta")

    # Save each record separately
    count = 0
    for record in records:
        file_path = os.path.join(output_dir, f"{record.id}.fasta")
        with open(file_path, "w") as f:
            SeqIO.write(record, f, "fasta")
        print(f"File saved: {file_path}")

        count += 1
        # Introduce a 1-second delay after every 3 sequences
        # To abide by the rules of downloading from ncbi
        if count % 3 == 0:
            time.sleep(1)

    fetch_handle.close()
    print(f"Download complete! All FASTA files are saved in '{output_dir}'.")


if __name__ == "__main__":
    # Timer
    start_time = time.time()

    # Accession lists
    denv1 = '/home/ibab/Desktop/DENV_serotyping/accession1.txt'
    denv2 = '/home/ibab/Desktop/DENV_serotyping/accession2.txt'
    denv3 = '/home/ibab/Desktop/DENV_serotyping/accession3.txt'
    denv4 = '/home/ibab/Desktop/DENV_serotyping/accession4.txt'

    # List of lists
    master = [denv1, denv2, denv3, denv4]

    # Email for Entrez
    myemail = "nimalanmadhavan@gmail.com"

    # Define the output directories for each variant type
    output_dirs = [
        '/home/ibab/Desktop/DENV_serotyping/data/wgs/DENV-1',
        '/home/ibab/Desktop/DENV_serotyping/data/wgs/DENV-2',
        '/home/ibab/Desktop/DENV_serotyping/data/wgs/DENV-3',
        '/home/ibab/Desktop/DENV_serotyping/data/wgs/DENV-4'
    ]

    # Ensure output directories exist
    for directory in output_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)

    for i in range(len(master)):
        process_text(master[i], myemail, output_dirs[i])

    neg_virus = ['SARS-CoV-2', 'ZV', 'CKGV', 'JEV', 'HPCV', 'WNV']
    root_dir = '/home/ibab/Desktop/DENV_serotyping/data/negative_class'

    for virus in neg_virus:
        search_term = virus + ' complete genome'
        vir_dir = root_dir + '/' + virus
        download_ncbi_fasta(search_term, 100, vir_dir, myemail)

    ksv_dir = root_dir + '/KSV'
    download_ncbi_fasta('Kyasanur Forest virus complete genome', 50, ksv_dir, myemail)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Time taken for execution: {execution_time} seconds.')
