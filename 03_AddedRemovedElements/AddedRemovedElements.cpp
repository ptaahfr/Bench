#include <chrono>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <set>
#include <unordered_set>

std::default_random_engine randomEngine((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());

template <typename ELEM_TYPE>
std::vector<ELEM_TYPE> generate_random_permutation(size_t count)
{
    std::vector<ELEM_TYPE> result(count);
    std::uniform_int_distribution<ELEM_TYPE> distrib(std::numeric_limits<ELEM_TYPE>::min(), std::numeric_limits<ELEM_TYPE>::max());

    std::generate(result.begin(), result.end(), [&] { return distrib(randomEngine); });

    return result;
}

template <typename ELEM_TYPE>
std::pair<size_t, size_t> brute_force(std::vector<ELEM_TYPE> const & array1, std::vector<ELEM_TYPE> const & array2)
{
    size_t added = 0;
    size_t removed = 0;

    for (auto const & elt1 : array1)
    {
        removed++;
        for (auto const & elt2 : array2)
        {
            if (elt1 != elt2)
            {
                continue;
            }

            removed--;
            break;
        }
    }

    for (auto const & elt2 : array2)
    {
        added++;
        for (auto const & elt1 : array1)
        {
            if (elt1 != elt2)
            {
                continue;
            }

            added--;
            break;
        }
    }

    return std::make_pair(removed, added);
}

template <typename ELEM_TYPE>
std::pair<size_t, size_t> using_bitfield(std::vector<ELEM_TYPE> const & array1, std::vector<ELEM_TYPE> const & array2)
{
    auto const minMax1(std::minmax_element(array1.begin(), array1.end()));
    auto const minMax2(std::minmax_element(array2.begin(), array2.end()));

    auto minValue(std::min(*std::get<0>(minMax1), *std::get<0>(minMax2)));
    auto maxValue(std::max(*std::get<1>(minMax1), *std::get<1>(minMax2)));

    size_t const bitPerElem(sizeof(uintptr_t) * 8);
    size_t const log2BitPerElem(sizeof(uintptr_t) == 8 ? 6 : 5);
    size_t const maskLow(bitPerElem - 1);

    std::vector<uintptr_t> bitfield1((maxValue - minValue + bitPerElem) / bitPerElem);
    std::vector<uintptr_t> bitfield2((maxValue - minValue + bitPerElem) / bitPerElem);

    size_t added = 0;
    size_t removed = 0;

    for (auto const & elt : array1)
    {
        auto const low((elt - minValue) & maskLow);
        auto const high((elt - minValue) >> log2BitPerElem);

        bitfield1[high] |= (1 << low);
    }

    for (auto const & elt : array2)
    {
        auto const low((elt - minValue) & maskLow);
        auto const high((elt - minValue) >> log2BitPerElem);

        if (0 == (bitfield1[high] & (1 << low)))
        {
            added++;
        }

        bitfield2[high] |= (1 << low);
    }

    for (auto const & elt : array1)
    {
        auto const low((elt - minValue) & maskLow);
        auto const high((elt - minValue) >> log2BitPerElem);

        if (0 == (bitfield2[high] & (1 << low)))
        {
            removed++;
        }
    }

    return std::make_pair(removed, added);
}

template <typename ELEM_TYPE, typename SET_TYPE = std::set<ELEM_TYPE> >
std::pair<size_t, size_t> using_set(std::vector<ELEM_TYPE> const & array1, std::vector<ELEM_TYPE> const & array2)
{
    size_t added = 0;
    size_t removed = 0;

    SET_TYPE tempSet1, tempSet2;


    for (auto const & elt1 : array1)
    {
        tempSet1.insert(elt1);
    }

    for (auto const & elt2 : array2)
    {
        if (tempSet1.find(elt2) == tempSet1.end())
            added++;
        tempSet2.insert(elt2);
    }

    for (auto const & elt1 : array1)
    {
        if (tempSet2.find(elt1) == tempSet2.end())
            removed++;
    }

    return std::make_pair(removed, added);
}

template <typename FUNC, typename ELEM_TYPE>
void test_one(char const * label, FUNC && f, std::vector<ELEM_TYPE> const & array1, std::vector<ELEM_TYPE> const & array2)
{
    std::cout << "Testing (" << array1.size() << " elements) : " << label << "..." << std::endl;
    auto beginTp(std::chrono::high_resolution_clock::now());
    auto res(f(array1, array2));
    auto endTp(std::chrono::high_resolution_clock::now());
    std::cout << "Result: " << std::get<0>(res) << " removed, " << std::get<1>(res) << " added" << std::endl;
    std::cout << "took " << std::chrono::duration_cast<std::chrono::microseconds>(endTp - beginTp).count() << " microseconds" << std::endl;
    std::cout << std::endl;
}

template <typename ELEM_TYPE>
void test_all(size_t count)
{
    auto array1(generate_random_permutation<ELEM_TYPE>(count));
    auto array2(generate_random_permutation<ELEM_TYPE>(count));

    test_one("using_set", using_set<ELEM_TYPE>, array1, array2);
    test_one("using_unordered_set", using_set<ELEM_TYPE, std::unordered_set<ELEM_TYPE> >, array1, array2);
    test_one("using_bitfield", using_bitfield<ELEM_TYPE>, array1, array2);
    test_one("brute_force", brute_force<ELEM_TYPE>, array1, array2);
}

int main()
{
    test_all<uint16_t>(1<<15);
}