"""
Generate synthetic person records with controlled duplication for
entity-resolution evaluation.
"""

import argparse
import csv
import random

from faker import Faker

FIELDS = ('first_name', 'last_name', 'birthdate', 'address', 'phone_number')


def perturb_record(record, fake):
    """Copy of record with one field minimally corrupted."""
    field_to_change = random.randint(0, len(record) - 1)
    perturbed_record = list(record)

    if field_to_change == 0 or field_to_change == 1:
        # drop a leading or trailing char from the name
        original = record[field_to_change] or ""
        if len(original) > 1:
            perturbed_record[field_to_change] = random.choice(
                [original[1:], original[:-1]]
            )
    elif field_to_change == 2:
        # different date of birth
        perturbed_record[field_to_change] = fake.date_of_birth(minimum_age=18, maximum_age=100)
    elif field_to_change == 3:
        # mangle address punctuation
        perturbed_record[field_to_change] = record[field_to_change].replace(",", ";")
    else:
        # truncate phone number
        perturbed_record[field_to_change] = record[field_to_change][:-1]

    return tuple(perturbed_record)


def generate_synthetic_data(
    output_file,
    num_records,
    duplicate_percentage,
    close_match_percentage,
    close_nonmatch_percentage,
    seed=0,
):
    """
    Build a synthetic person-record dataset and write it to output_file.

    Each unique record gets a fresh group_id. Duplicates and close-matches
    inherit the source record's group_id, so group_id is the supervised
    signal for MRR. Close non-matches get singleton group_ids and act as
    hard negatives.
    """
    fake = Faker()
    Faker.seed(seed)
    random.seed(seed)

    unique_percentage = 1 - duplicate_percentage - close_match_percentage - close_nonmatch_percentage
    num_unique_records = int(num_records * unique_percentage)

    # all records carry (group_id, fields_tuple)
    next_group_id = 1
    unique_records = []
    for _ in range(num_unique_records):
        record = (
            fake.first_name(),
            fake.last_name(),
            fake.date_of_birth(minimum_age=18, maximum_age=100),
            fake.address().replace('\n', ' '),
            fake.phone_number(),
        )
        unique_records.append((next_group_id, record))
        next_group_id += 1

    # duplicates: exact copies of a unique record, same group_id
    duplicates = []
    n_dupes = int(num_records * duplicate_percentage)
    for _ in range(n_dupes):
        gid, rec = random.choice(unique_records)
        duplicates.append((gid, rec))

    # close matches: perturbed copies of a unique record, same group_id
    close_matches = []
    n_close = int(num_records * close_match_percentage)
    for _ in range(n_close):
        gid, rec = random.choice(unique_records)
        close_matches.append((gid, perturb_record(rec, fake)))

    # close non-matches: perturbed records that don't correspond to
    # any other record. each gets its own singleton group_id.
    close_nonmatches = []
    seen_records = {rec for _, rec in unique_records + duplicates + close_matches}
    n_nonmatch = int(num_records * close_nonmatch_percentage)
    while len(close_nonmatches) < n_nonmatch:
        _, source = random.choice(unique_records)
        rec = perturb_record(source, fake)
        if rec in seen_records:
            continue
        seen_records.add(rec)
        close_nonmatches.append((next_group_id, rec))
        next_group_id += 1

    all_records = unique_records + duplicates + close_matches + close_nonmatches
    random.shuffle(all_records)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(('id', 'group_id') + FIELDS)
        for new_id, (group_id, record) in enumerate(all_records, start=1):
            writer.writerow((new_id, group_id) + record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic person records.')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file')
    parser.add_argument('-n', '--num_records', type=int, required=True, help='Number of records to generate')
    parser.add_argument('-d', '--duplicate_percentage', type=float, required=True, help='Percentage of duplicate records')
    parser.add_argument('-c', '--close_match_percentage', type=float, required=True, help='Percentage of close match records')
    parser.add_argument('-nc', '--close_nonmatch_percentage', type=float, required=True, help='Percentage of close non-match records')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default 0)')

    args = parser.parse_args()

    if args.duplicate_percentage + args.close_match_percentage + args.close_nonmatch_percentage > 1:
        raise ValueError("The sum of duplicate_percentage, close_match_percentage, and close_nonmatch_percentage should not exceed 1.")

    generate_synthetic_data(
        args.output,
        args.num_records,
        args.duplicate_percentage,
        args.close_match_percentage,
        args.close_nonmatch_percentage,
        seed=args.seed,
    )
