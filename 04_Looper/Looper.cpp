// Really needed by the example
#include <type_traits>
#include <tuple>

// Alias to make code more compact
template <size_t N>
using ConstSizeT = std::integral_constant<size_t, N>;

// 1) Starting point, simple 1D loop
template <typename BOUND, typename FUNC>
void For1D(BOUND min, BOUND max, FUNC && func)
{
    for (BOUND x = min; x < max; ++x)
    {
        func(x);
    }
}

// Used only for std::cout
#include <iostream>

void test_looper_for_simple()
{
    For1D(0, 10, [](size_t x) {std::cout << x << std::endl; });
    std::cout << std::endl;
}

// 2) More flexibility by adding passthrough arguments
template <typename BOUND, typename FUNC, typename... ARGS>
void For1D(BOUND min, BOUND max, FUNC && func, ARGS &&... args)
{
    for (BOUND x = min; x < max; ++x)
    {
        func(x, std::forward<ARGS>(args)...);
    }
}

void test_looper_for_args()
{
    For1D(0, 10, [](size_t x, std::ostream & os)
    {
        os << x << std::endl;
    }, std::cout);
    std::cout << std::endl;
}

// 3) 1D loop specialization when bounds are compile time constants (unrolling)

// Recursion termination (BOUND_MIN == BOUND_MAX)
template <typename INDEX, INDEX BOUND_MAX, typename FUNC, typename... ARGS>
void For1D(std::integral_constant<INDEX, BOUND_MAX> boundMin, std::integral_constant<INDEX, BOUND_MAX> boundMax, FUNC && func, ARGS &&... args)
{
}

// Unroll by calling recursively the functions
template <typename INDEX, INDEX BOUND_MIN, INDEX BOUND_MAX, typename FUNC, typename... ARGS>
void For1D(std::integral_constant<INDEX, BOUND_MIN> boundMin, std::integral_constant<INDEX, BOUND_MAX> boundMax, FUNC && func, ARGS &&... args)
{
    func(boundMin, std::forward<ARGS>(args)...);
    For1D(std::integral_constant<INDEX, BOUND_MIN + 1>(), boundMax, std::forward<FUNC>(func), std::forward<ARGS>(args)...);
}

void test_looper_unrolled()
{
    auto const xValues = std::make_tuple((void const *)"Zero", 1.0, "Two", 3, 4.0f);
    For1D(ConstSizeT<0>(), std::tuple_size<decltype(xValues)>(), [&](auto x, std::ostream & os)
    {
        // At each iteration x is a different type mapping to a compile time constant,
        // For instance, we can use it to enumerate a tuple with a different type of object at each step
        os << x << " => " << std::get<decltype(x)::value>(xValues) << std::endl;
    }, std::cout);
    std::cout << std::endl;
}

// 4) 2D case allowing permutation of loops, we use tuple for bounds

// Class to define the 2D permutation
template <size_t X0, size_t X1>
class Permutation2D
{
    static_assert(X0 + X1 == 1 && X0 < 2 && X1 < 2, "Invalid permutation");
};

template <size_t X0, size_t X1, typename BOUND0, typename BOUND1, typename FUNC, typename... ARGS>
void For2D(Permutation2D<X0, X1> permutation, std::tuple<BOUND0, BOUND1> const & boundsMin, std::tuple<BOUND0, BOUND1> const & boundsMax, FUNC && func, ARGS &&... args)
{
    for (BOUND0 y = std::get<X0>(boundsMin); y < std::get<X0>(boundsMax); ++y)
    {
        for (BOUND1 x = std::get<X1>(boundsMin); x < std::get<X1>(boundsMax); ++x)
        {
            auto const indices = std::make_tuple(x, y);
            func(std::get<X0>(indices), std::get<X1>(indices), std::forward<ARGS>(args)...);
        }
    }
}

void test_looper_for_permuted_2d()
{
    For2D(Permutation2D<1, 0>(), std::make_tuple(0, 0), std::make_tuple(4, 2), [](size_t x, size_t y, std::ostream & os)
    {
        os << x << "," << y << std::endl;
    }, std::cout);
    std::cout << std::endl;
}


// Utility function to insert a new element into a tuple
namespace detailsInsertTuple
{
    template <size_t AT, typename WHAT, typename... ELEMENTS>
    auto InsertTuple(ConstSizeT<sizeof...(ELEMENTS)> iteration, ConstSizeT<AT> at, WHAT && what, std::tuple<ELEMENTS...> const & elements)
    {
        return std::make_tuple();
    }

    template <size_t ITERATION, typename WHAT, typename... ELEMENTS>
    auto InsertTuple(ConstSizeT<ITERATION> iteration, ConstSizeT<ITERATION> at, WHAT && what, std::tuple<ELEMENTS...> const & elements);

    template <size_t ITERATION, size_t AT, typename WHAT, typename... ELEMENTS, std::enable_if_t<(ITERATION < sizeof...(ELEMENTS)), void *> = nullptr>
        auto InsertTuple(ConstSizeT<ITERATION> iteration, ConstSizeT<AT> at, WHAT && what, std::tuple<ELEMENTS...> const & elements);

    template <size_t ITERATION, typename WHAT, typename... ELEMENTS>
    auto InsertTuple(ConstSizeT<ITERATION> iteration, ConstSizeT<ITERATION> at, WHAT && what, std::tuple<ELEMENTS...> const & elements)
    {
        return std::tuple_cat(std::make_tuple(what), InsertTuple(ConstSizeT<ITERATION + 1>(), at, std::forward<WHAT>(what), elements));
    }

    template <size_t ITERATION, size_t AT, typename WHAT, typename... ELEMENTS, std::enable_if_t<(ITERATION < sizeof...(ELEMENTS)), void *>>
    auto InsertTuple(ConstSizeT<ITERATION> iteration, ConstSizeT<AT> at, WHAT && what, std::tuple<ELEMENTS...> const & elements)
    {
        return std::tuple_cat(std::make_tuple(std::get<ITERATION>(elements)), InsertTuple(ConstSizeT<ITERATION + 1>(), at, std::forward<WHAT>(what), elements));
    }
}

template <size_t AT, typename WHAT, typename... ELEMENTS>
auto InsertTuple(ConstSizeT<AT> at, WHAT && what, std::tuple<ELEMENTS...> const & elements)
{
    return detailsInsertTuple::InsertTuple(ConstSizeT<0>(), at, std::forward<WHAT>(what), elements);
}

// Utilities used only to make sure the permutation validity is checked at compile time
namespace details
{
    template <size_t X0, size_t... XN>
    class ExpectedPermutationSum : public std::integral_constant<size_t, ExpectedPermutationSum<XN...>::value + sizeof...(XN)>
    {
    };

    template <size_t XN>
    class ExpectedPermutationSum<XN> : public std::integral_constant<size_t, 0>
    {

    };

    template <size_t X0, size_t... XN>
    class SumOfElements : public std::integral_constant<size_t, SumOfElements<XN...>::value + X0>
    {
    };

    template <size_t X0>
    class SumOfElements<X0> : public std::integral_constant<size_t, X0>
    {
    };

    template <size_t MAX_VALUE, size_t X0, size_t... XN>
    class AreValuesUnder : public std::integral_constant<bool, (X0 < MAX_VALUE) && AreValuesUnder<MAX_VALUE, XN...>::value>
    {
    };

    template <size_t MAX_VALUE, size_t X0>
    class AreValuesUnder<MAX_VALUE, X0> : public std::integral_constant<bool, (X0 < MAX_VALUE)>
    {
    };
}

// 5) ND case allowing permutation for loops
template <size_t... XN>
class Permutation
{
    static_assert(details::SumOfElements<XN...>::value == details::ExpectedPermutationSum<XN...>::value
               && details::AreValuesUnder<sizeof...(XN), XN...>::value, "Invalid permutation");
};


namespace details
{
    // Function to call the function unpacking the indices (we don't want to call f(std::tuple(x, y, z), ...) but f(x, y, z, ...)
    template <size_t... SEQ, typename... INDICES, typename FUNC, typename... ARGS>
    static void CallFunc(std::index_sequence<SEQ...>, std::tuple<INDICES...> const & indices, FUNC && func, ARGS &&... args)
    {
        func(std::get<SEQ>(indices)..., std::forward<ARGS>(args)...);
    }

    // End of the recursion (LOOP_LEVEL == MAX_LOOP_LEVEL)
    template <size_t MAX_LOOP_LEVEL, size_t... XN, typename... BOUNDS, typename FUNC, typename... ARGS>
    static void ForNSimple(ConstSizeT<MAX_LOOP_LEVEL> loopLevel, ConstSizeT<MAX_LOOP_LEVEL> maxLoopLevel, std::tuple<BOUNDS...> const & indices, Permutation<XN...> permutation,
        std::tuple<BOUNDS...> const & boundsMin, std::tuple<BOUNDS...> const & boundsMax, FUNC && func, ARGS &&... args)
    {
        CallFunc(std::make_index_sequence<sizeof...(XN)>(), indices, std::forward<FUNC>(func), std::forward<ARGS>(args)...);
    }

    // Recursive definition, we build at each step the indices tuple and increase the loopLevel
    // At each level we enumerate index and insert it at the right position in the "indices" tuple.
    template <size_t LOOP_LEVEL, size_t MAX_LOOP_LEVEL, size_t... XN, typename... BOUNDS, typename FUNC, typename... ARGS>
    static void ForNSimple(ConstSizeT<LOOP_LEVEL> loopLevel, ConstSizeT<MAX_LOOP_LEVEL> maxLoopLevel, std::tuple<BOUNDS...> const & indices, Permutation<XN...> permutation,
        std::tuple<BOUNDS...> const & boundsMin, std::tuple<BOUNDS...> const & boundsMax, FUNC && func, ARGS &&... args)
    {
        for (auto index = std::get<LOOP_LEVEL>(boundsMin); index < std::get<LOOP_LEVEL>(boundsMax); ++index)
        {
            ForNSimple(ConstSizeT<LOOP_LEVEL + 1>(), maxLoopLevel, InsertTuple(loopLevel, index, indices), permutation,
                boundsMin, boundsMax, std::forward<FUNC>(func), std::forward<ARGS>(args)...);
        }
    }
}

// Entry point, simply calls the recursive function adding arguments needed for recursive state
// We initialize indices with boundsMin
template <size_t... XN, typename... BOUNDS, typename FUNC, typename... ARGS>
void ForNSimple(Permutation<XN...> permutation, std::tuple<BOUNDS...> const & boundsMin, std::tuple<BOUNDS...> const & boundsMax, FUNC && func, ARGS &&... args)
{
    details::ForNSimple(ConstSizeT<0>(), ConstSizeT<sizeof...(XN)>(), boundsMin, permutation, boundsMin, boundsMax, std::forward<FUNC>(func), std::forward<ARGS>(args)...);
}

void test_looper_for_permuted_nd()
{
    ForNSimple(Permutation<1, 0, 3, 2>(), std::make_tuple(0, 0, 0, 0), std::make_tuple(4, 2, 3, 5), [](size_t x, size_t y, size_t z, size_t w, std::ostream & os)
    {
        os << x << "," << y << "," << z << "," << w << std::endl;
    }, std::cout);
    std::cout << std::endl;
}

// 6) ND case allowing permutation for loops, using For1D so we draw also For1D specializations like compile-time constants unrolling
namespace details
{
    // End of the recursion (LOOP_LEVEL == MAX_LOOP_LEVEL)
    template <size_t MAX_LOOP_LEVEL, typename... INDICES, size_t... XN, typename... BOUNDS_MIN, typename... BOUNDS_MAX, typename FUNC, typename... ARGS>
    static void ForN(ConstSizeT<MAX_LOOP_LEVEL> loopLevel, ConstSizeT<MAX_LOOP_LEVEL> maxLoopLevel, std::tuple<INDICES...> const & indices, Permutation<XN...> permutation,
        std::tuple<BOUNDS_MIN...> const & boundsMin, std::tuple<BOUNDS_MAX...> const & boundsMax, FUNC && func, ARGS &&... args)
    {
        CallFunc(std::make_index_sequence<sizeof...(XN)>(), indices, std::forward<FUNC>(func), std::forward<ARGS>(args)...);
    }

    // We need to declare this before the following lambda.
    template <size_t LOOP_LEVEL, size_t MAX_LOOP_LEVEL, typename... INDICES, size_t... XN, typename... BOUNDS_MIN, typename... BOUNDS_MAX, typename FUNC, typename... ARGS>
    static void ForN(ConstSizeT<LOOP_LEVEL> loopLevel, ConstSizeT<MAX_LOOP_LEVEL> maxLoopLevel, std::tuple<INDICES...> const & indices, Permutation<XN...> permutation,
        std::tuple<BOUNDS_MIN...> const & boundsMin, std::tuple<BOUNDS_MAX...> const & boundsMax, FUNC && func, ARGS &&... args);

    // "Lambda" (as a function because we need variadic args...) to call the for recursion
    class ForNToFor1D
    {
    public:
        template <typename INDEX, size_t LOOP_LEVEL, size_t MAX_LOOP_LEVEL, typename... INDICES, size_t... XN, typename... BOUNDS_MIN, typename... BOUNDS_MAX, typename FUNC, typename... ARGS>
        void operator()(INDEX index, ConstSizeT<LOOP_LEVEL> loopLevel, ConstSizeT<MAX_LOOP_LEVEL> maxLoopLevel, std::tuple<INDICES...> const & indices, Permutation<XN...> permutation,
            std::tuple<BOUNDS_MIN...> const & boundsMin, std::tuple<BOUNDS_MAX...> const & boundsMax, FUNC && func, ARGS &&... args) const
        {
            ForN(ConstSizeT<LOOP_LEVEL + 1>(), maxLoopLevel, InsertTuple(loopLevel, index, indices),
                permutation, boundsMin, boundsMax, std::forward<FUNC>(func), std::forward<ARGS>(args)...);
        }
    };

    // Recursive definition, we build at each step the indices tuple and increase the loopLevel
    // This one using For1D so we can benefit For1D specialization, we need to pass by a functor (above), because this is C++14 and variadic are not supported in lambdas
    template <size_t LOOP_LEVEL, size_t MAX_LOOP_LEVEL, typename... INDICES, size_t... XN, typename... BOUNDS_MIN, typename... BOUNDS_MAX, typename FUNC, typename... ARGS>
    static void ForN(ConstSizeT<LOOP_LEVEL> loopLevel, ConstSizeT<MAX_LOOP_LEVEL> maxLoopLevel, std::tuple<INDICES...> const & indices, Permutation<XN...> permutation,
        std::tuple<BOUNDS_MIN...> const & boundsMin, std::tuple<BOUNDS_MAX...> const & boundsMax, FUNC && func, ARGS &&... args)
    {
        For1D(std::get<LOOP_LEVEL>(boundsMin), std::get<LOOP_LEVEL>(boundsMax), ForNToFor1D(),
            loopLevel, maxLoopLevel, indices, permutation, boundsMin, boundsMax, std::forward<FUNC>(func), std::forward<ARGS>(args)...);
    }
}

// Entry point, simply calls the recursive function adding arguments needed for recursive state
template <size_t... XN, typename... BOUNDS_MIN, typename... BOUNDS_MAX, typename FUNC, typename... ARGS>
void ForN(Permutation<XN...> permutation, std::tuple<BOUNDS_MIN...> const & boundsMin, std::tuple<BOUNDS_MAX...> const & boundsMax, FUNC && func, ARGS &&... args)
{
    details::ForN(ConstSizeT<0>(), ConstSizeT<sizeof...(XN)>(), boundsMin, permutation, boundsMin, boundsMax, std::forward<FUNC>(func), std::forward<ARGS>(args)...);
}

void test_looper_for_permuted_nd_withunroll()
{
    auto const wValues = std::make_tuple(0, "One", 2, "Three", "Four", 5);

    ForN(Permutation<2, 0, 1>(), std::make_tuple(0, 0, ConstSizeT<0>()), std::make_tuple(2, 3, ConstSizeT<6>()), [=](size_t x, size_t y, auto w, std::ostream & os)
    {
        // Here w type is different at each iteration because of static unrolling so we use it in a heterogeneous tuple
        os << x << "," << y << "," << std::get<decltype(w)::value>(wValues) << std::endl;
    }, std::cout);

    ForN(Permutation<1, 2, 0>(), std::make_tuple(0, 0, ConstSizeT<0>()), std::make_tuple(2, 3, ConstSizeT<6>()), [=](size_t x, size_t y, auto w, std::ostream & os)
    {
        // Here w type is different at each iteration because of static unrolling so we use it in a heterogeneous tuple
        os << x << "," << y << "," << std::get<decltype(w)::value>(wValues) << std::endl;
    }, std::cout);
}

int main()
{
    test_looper_for_simple();
    std::cout << std::endl;

    test_looper_for_args();
    std::cout << std::endl;

    test_looper_unrolled();
    std::cout << std::endl;

    test_looper_for_permuted_2d();
    std::cout << std::endl;

    test_looper_for_permuted_nd();
    std::cout << std::endl;

    test_looper_for_permuted_nd_withunroll();
    std::cout << std::endl;
}