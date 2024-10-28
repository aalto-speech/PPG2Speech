# Check if the first column contains "test" and write to the appropriate file
{
    if ($1 ~ /test/) {
        print $0 > parent_folder "/test/" filename
    } else {
        print $0 > parent_folder "/train_val/" filename
    }
}