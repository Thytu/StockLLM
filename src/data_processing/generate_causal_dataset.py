from .get_random_data_samples import get_random_data_samples


def main(
    number_of_training_samples: int,
    number_of_testing_samples: int,
    path_to_output_dataset: str,
) -> None:
    
    total_number_of_samples = number_of_training_samples + number_of_testing_samples

    samples = get_random_data_samples(
        number_of_samples=total_number_of_samples,
    )
    
    dataset_dict = samples.train_test_split(
        test_size=number_of_testing_samples / total_number_of_samples,
    )
    
    dataset_dict.save_to_disk(path_to_output_dataset)
