#include <mpi.h>
#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <stdexcept>

class ParallelCharCounter {
    static constexpr int ALPHABET_SIZE = 26;
    int rank_;
    int size_;
    std::vector<char> input_chars_;
    std::vector<int> global_counts_;
    double elapsed_time_;

    void validate_input_size(int n) const {
        if (n <= 0) {
            throw std::runtime_error("Input size must be positive");
        }
    }

    void read_and_broadcast_input() {
        int n = 0;
        if (rank_ == 0) {
            std::cout << "Enter the number of characters and characters: ";
            std::cin >> n;
            validate_input_size(n);

            input_chars_.resize(n);
            std::cin >> input_chars_.data();
        }

        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank_ != 0) {
            input_chars_.resize(n);
        }
        MPI_Bcast(input_chars_.data(), n, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    std::vector<int> compute_local_counts() const {
        std::vector<int> local_counts(ALPHABET_SIZE, 0);
        for (size_t i = rank_; i < input_chars_.size(); i += size_) {
            char c = input_chars_[i];
            if (c >= 'a' && c <= 'z') {
                local_counts[c - 'a']++;
            }
        }
        return local_counts;
    }

    void reduce_counts(const std::vector<int>& local_counts) {
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
    ParallelCharCounter(int argc, char* argv[]) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    ~ParallelCharCounter() {
        MPI_Finalize();
    }

    void run() {
        auto start_time = MPI_Wtime();

        try {
            read_and_broadcast_input();
            auto local_counts = compute_local_counts();
            reduce_counts(local_counts);
            
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
    ParallelCharCounter counter(argc, argv);
    counter.run();
    return 0;
}