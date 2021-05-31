int is_prime(const int p)
{
    for (int i = 3; i <= sqrt(p); i++)
    {
        if (p % i == 0)
        {
            return 0;
        }
    }
    return 1;
}

int goldbach(int number, int rank, int first)
{
    int i = 2;
    for (int j = number - i; j > 2; j--, i++)
    {
        if (is_prime(i) == 1 && is_prime(j) == 1)
        {
            if (first)
            {
                printf("[Rank %d] The first sum is %d + %d = %d \n", rank, i, j, number);
            }
            return 1;
        }
    }
    return 0;
}

int call_goldbach_range(int start, int end, int rank)
{
    int result;

    for (int i = start; i <= end; i++)
    {
        if (goldbach(i, rank, i == start) == 0)
        {
            return 0;
        }
    }
    return 1;
}