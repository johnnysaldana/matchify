import csv
import random
from faker import Faker
import argparse


def perturb_record(record, fake):
    field_to_change = random.randint(0, len(record) - 1)
    perturbed_record = list(record)

    if field_to_change == 0:
        perturbed_record[field_to_change] = random.choice([record[field_to_change][1:], record[field_to_change][:-1]])
    elif field_to_change == 1:
        perturbed_record[field_to_change] = random.choice([record[field_to_change][1:], record[field_to_change][:-1]])
    elif field_to_change == 2:
        perturbed_record[field_to_change] = fake.date_of_birth(minimum_age=18, maximum_age=100)
    elif field_to_change == 3:
        perturbed_record[field_to_change] = record[field_to_change].replace(",", ";")
    else:
        perturbed_record[field_to_change] = record[field_to_change][:-1]

    return tuple(perturbed_record)


def generate_synthetic_data(output_file, num_records, duplicate_percentage, close_match_percentage, close_nonmatch_percentage):
    fake = Faker()
    Faker.seed(0)  # Seed to ensure reproducibility

    # Calculate the number of unique records
    unique_percentage = 1 - duplicate_percentage - close_match_percentage - close_nonmatch_percentage
    num_unique_records = int(num_records * unique_percentage)

    unique_records = []

    for _ in range(num_unique_records):
        record = (
            fake.first_name(),
            fake.last_name(),
            fake.date_of_birth(minimum_age=18, maximum_age=100),
            fake.address().replace('\n', ' '),
            fake.phone_number()
        )
        unique_records.append(record)

    # Generate duplicates
    duplicates = random.choices(unique_records, k=int(num_records * duplicate_percentage))

    # Generate close matches
    close_matches = [perturb_record(record, fake) for record in random.choices(unique_records, k=int(num_records * close_match_percentage))]

    # Generate close non-matches
    close_nonmatches = []
    while len(close_nonmatches) < int(num_records * close_nonmatch_percentage):
        record = perturb_record(random.choice(unique_records), fake)
        if record not in unique_records + duplicates + close_matches + close_nonmatches:
            close_nonmatches.append(record)
    # Combine and shuffle the records
    all_records = unique_records + duplicates + close_matches + close_nonmatches
    random.shuffle(all_records)

    # Write the records to a CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['first_name', 'last_name', 'birthdate', 'address', 'phone_number'])
        for record in all_records:
            writer.writerow(record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic person records.')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file')
    parser.add_argument('-n', '--num_records', type=int, required=True, help='Number of records to generate')
    parser.add_argument('-d', '--duplicate_percentage', type=float, required=True, help='Percentage of duplicate records')
    parser.add_argument('-c', '--close_match_percentage', type=float, required=True, help='Percentage of close match records')
    parser.add_argument('-nc', '--close_nonmatch_percentage', type=float, required=True, help='Percentage of close non-match records')

    args = parser.parse_args()

    if args.duplicate_percentage + args.close_match_percentage + args.close_nonmatch_percentage > 1:
        raise ValueError("The sum of duplicate_percentage, close_match_percentage, and close_nonmatch_percentage should not exceed 1.")

    generate_synthetic_data(args.output, args.num_records, args.duplicate_percentage, args.close_match_percentage, args.close_nonmatch_percentage)
