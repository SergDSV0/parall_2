#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <stdexcept>

class CharacterFrequencyCounter {
    static constexpr int ALPHABET_SIZE = 26;
    int rank_;
    int size_;
    std::string input_str_;
    std::vector<int> global_counts_;
    double elapsed_time_;

    void validate_environment() const {
        if (size_ < 1) {
            throw std::runtime_error("At least one process required");
        }
    }

    void master_read_input() {
        int n;
        std::cout << "Enter the number of characters and characters: ";
        std::cin >> n >> input_str_;

        if (input_str_.length() != static_cast<size_t>(n)) {
            throw std::runtime_error("Input length mismatch");
        }

        // Broadcast string to all workers
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<char*>(input_str_.data()), n, 
                 MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    void worker_receive_input() {
        int n;
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        input_str_.resize(n);
        MPI_Bcast(const_cast<char*>(input_str_.data()), n, 
                 MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    std::vector<int> count_local_frequencies() const {
        std::vector<int> local_counts(ALPHABET_SIZE, 0);
        for (size_t i = rank_; i < input_str_.size(); i += size_) {
            char c = tolower(input_str_[i]);
            if (c >= 'a' && c <= 'z') {
                local_counts[c - 'a']++;
            }
        }
        return local_counts;
    }

    void gather_results(const std::vector<int>& local_counts) {
        if (rank_ == 0) {
            global_counts_.resize(ALPHABET_SIZE);
        }
        MPI_Reduce(local_counts.data(), global_counts_.data(), 
                  ALPHABET_SIZE, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    void print_results() const {
        if (rank_ != 0) return;

        std::cout << "Character frequencies:\n";
        for (int i = 0; i < ALPHABET_SIZE; ++i) {
            if (global_counts_[i] > 0) {
                std::cout << static_cast<char>('a' + i) 
                          << " = " << global_counts_[i] << "\n";
            }
        }
        std::cout << "Time taken: " << elapsed_time_ << " seconds\n";
    }

public:
    CharacterFrequencyCounter(int argc, char* argv[]) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    ~CharacterFrequencyCounter() {
        MPI_Finalize();
    }

    void run() {
        auto start_time = MPI_Wtime();

        try {
            validate_environment();

            if (rank_ == 0) {
                master_read_input();
            } else {
                worker_receive_input();
            }

            auto local_counts = count_local_frequencies();
            gather_results(local_counts);

            elapsed_time_ = MPI_Wtime() - start_time;
            print_results();

        } catch (const std::exception& e) {
            if (rank_ == 0) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        }
    }
};

int main(int argc, char* argv[]) {
    CharacterFrequencyCounter counter(argc, argv);
    counter.run();
    return 0;
}
