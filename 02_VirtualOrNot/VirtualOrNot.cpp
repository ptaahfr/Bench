
#include <iostream>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>

class Physio
{
public:
    ~Physio()
    {
    }

    double Poids;
    double Taille;
};

class Personne
{
public:
    ~Personne()
    {
    }

    int64_t UniqueId = std::rand();
    Physio P;
};

class PhysioVirtuel
{
public:
    virtual ~PhysioVirtuel()
    {
        // Au cas où
    }

    double Poids;
    double Taille;
};

class PersonneVirtuelle
{
public:
    virtual ~PersonneVirtuelle()
    {
        // Au cas où
    }

    int64_t UniqueId = std::rand();
    PhysioVirtuel P;
};

class PersonneGrosBouzin
{
public:
    ~PersonneGrosBouzin()
    {
    }

    int64_t UniqueId = std::rand();
    double Poids;
    double Taille;
};

class PersonneGrosBouzinVirtuel
{
public:
    virtual ~PersonneGrosBouzinVirtuel()
    {
        // Au cas où
    }

    int64_t UniqueId = std::rand();
    double Poids;
    double Taille;
};


int main()
{
    size_t count = 60000000;

#define TEST_FOR_CLASS(CLASS) \
    { \
        std::srand(0); \
        std::vector<CLASS> population(count); \
 \
        auto start = std::chrono::high_resolution_clock::now(); \
        std::sort(population.begin(), population.end(), [&](CLASS const & p1, CLASS const & p2) { return p1.UniqueId < p2.UniqueId; }); \
        auto end = std::chrono::high_resolution_clock::now(); \
 \
        std::cout << "sizeof(" << #CLASS << ") = " << sizeof(CLASS) << std::endl; \
        std::cout << "Timing [Not Virtual]: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0f << " ms" << std::endl; \
    }

    TEST_FOR_CLASS(Personne)
    TEST_FOR_CLASS(PersonneVirtuelle)
    TEST_FOR_CLASS(PersonneGrosBouzin)
    TEST_FOR_CLASS(PersonneGrosBouzinVirtuel)

    return 0;
}