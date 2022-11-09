#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <omp.h> 
#include <chrono>

using namespace std;

float calculate_distance(int x1, int y1, int x2, int y2);
vector<vector<int>> ComputeEuclideanDistanceMatrix(const vector<vector<int>>& locations, int n);
vector<vector<int>> read_tsp_file(const char* fname);
int firstMinInRow(int vertex_index);
int secondMinInRow(int i);
void TSPRec(int curr_bound, int current_weight, int level, std::vector<int> curr_pat, std::vector<int> vis);
void TSP();

int N;
std::vector<int> final_path;
int final_res = INT_MAX;
vector<vector<int>> distance_matrix = read_tsp_file("ulysses222.tsp.txt");

/* vector<vector<int>> distance_matrix = { {0,83,93,129,133,139,151,169,135,114,110},
                                        {83, 0,40,53,62,64,91, 116,93,84,95},
                                        {93,40, 0,42,42,49,59,81,54,44,58},
                                        {129,53,42, 0,11,11,46,72,65,70,88},
                                        {133,62,42,11, 0, 9,35,61,55,62,82},
                                        {139,64,49,11, 9, 0,39,65,63,71,90},
                                        {151,91,59,46,35,39, 0,26,34,52,71},
                                        {169, 116,81,72,61,65,26, 0,37,59,75},
                                        {135,93,54,65,55,63,34,37, 0,22,39},
                                        {114,84,44,70,62,71,52,59,22, 0,20},
                                        {110,95,58,88,82,90,71,75,39,20, 0} };*/

void copyToFinal(std::vector<int>& current_path) {
    int i;
    for (i = 0; i < N; i++)
        final_path.push_back(current_path[i]);
    final_path.push_back(current_path[0]);
}

//Function to find the minimum edge cost having an end at the vertex :vertex_index
//minimum value in selected row
//vertex_index - curr row
int firstMinInRow(int vertex_index) {
    int min = INT_MAX;
    for (int i = 0; i < distance_matrix.size(); i++)
    {
        if (distance_matrix[vertex_index][i] < min and vertex_index != i) {
            min = distance_matrix[vertex_index][i];
        }
    }
    return min;
}

//Function to find the minimum edge cost having an end at the vertex :i
//second minimum value in selected row
//vertex_index - curr row
int secondMinInRow(int i) {
    int first = INT_MAX, second = INT_MAX;

    for (int j = 0; j < distance_matrix.size(); j++) {

        if (i == j) {
            continue;
        }

        if (distance_matrix[i][j] <= first) {
            second = first;
            first = distance_matrix[i][j];
        }

        else if (distance_matrix[i][j] <= second && distance_matrix[i][j] != first) {
            second = distance_matrix[i][j];
        }

    }

    return second;

}

//function that takes as arguments:
//curr_bound -> lower bound of the root node - to co je výš ve stromu
//curr_weight-> stores the weight of the path so far - aktualni cost
//level-> current level while moving in the search space tree - level ve stromu
//curr_pat -> where the solution is being stored, which would later be copied to final_path
//vis -> stores info about which nodes were already visited
void TSPRec(int current_bound, int current_weight, int level, std::vector<int> curr_path, std::vector<int> visited) {
    if (level == N) {
        // check if there is an edge from last vertex in path back to the first vertex
        if (distance_matrix[curr_path[level - 1]][curr_path[0]] != 0) {
            int curr_res = current_weight + distance_matrix[curr_path[level - 1]][curr_path[0]];

#pragma omp critical
            if (curr_res < final_res) {
                copyToFinal(curr_path);
                final_res = curr_res;
            }
        }
        return;
    }

    int i;
#pragma omp parallel for default(none) \
    firstprivate(current_bound, current_weight, level) \
    shared(i,curr_path, visited,distance_matrix,final_res,N)
    for (i = 0; i < N; i++) {

        std::vector<int> visitedInPath{};
        std::vector<int> current_path{};
        int j;
        for (j = 0; j < N; j++) { visitedInPath.push_back(visited[j]); }
        for (j = 0; j < N; j++) { current_path.push_back(curr_path[j]); }

        if (distance_matrix[current_path[level - 1]][i] != 0 && visitedInPath[i] == 0) {
            int temp = current_bound;
            current_weight += distance_matrix[current_path[level - 1]][i];

            if (level == 1)
                current_bound -= ((firstMinInRow(current_path[level - 1]) +
                    firstMinInRow(i)) / 2);
            else
                current_bound -= ((secondMinInRow(current_path[level - 1]) +
                    firstMinInRow(i)) / 2);

            if (current_bound + current_weight < final_res) {
                current_path[level] = i;
                visitedInPath[i] = 1;
                // call TSPRec for the next level
                TSPRec(current_bound, current_weight, level + 1, current_path, visitedInPath);

            }

            //reset changes if not found better cost in branch at current level
            current_weight -= distance_matrix[current_path[level - 1]][i];
            current_bound = temp;
        }
    }
}

void TSP() {
    std::vector<int> current_path{};

    int current_bound = 0;
    std::vector<int> visitedInPath{};

    for (int i = 0; i < N + 1; i++) current_path.push_back(-1);
    for (int i = 0; i < N; i++) visitedInPath.push_back(0);


    int i;
#pragma omp parallel for reduction(+:current_bound)
    for (i = 0; i < N; i++)
        current_bound += (firstMinInRow(i) +
            secondMinInRow(i));

    current_bound = round(current_bound / 2);

    visitedInPath[0] = 1;
    current_path[0] = 0;
    TSPRec(current_bound, 0, 1, current_path, visitedInPath);
}

//euclidean distance between 2 points
float calculate_distance(int x1, int y1, int x2, int y2)
{
    // Calculating distance
    int res = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) * 1.0);
    return res;
}

// @brief Generate distance matrix.
vector<vector<int>> ComputeEuclideanDistanceMatrix(const vector<vector<int>>& locations, int n)
{
    vector<vector<int>> distances = vector<vector<int>>(n, vector<int>(n, int{ 0 }));
    for (int startingCity = 0; startingCity < n; startingCity++)
    {
        for (int endingCity = 0; endingCity < n; endingCity++)
        {
            if (startingCity != endingCity) {
                distances[startingCity][endingCity] = calculate_distance(locations[startingCity][0], locations[startingCity][1], locations[endingCity][0], locations[endingCity][1]);
            }
        }
    }
    return distances;
}

vector<vector<int>> read_tsp_file(const char* fname)
{
    ifstream file(fname);
    vector<int> xs, ys;
    vector<vector<int>> temp;
    vector<vector<int>> distance_matrix;

    if (file.is_open())
    {
        string line;

        getline(file, line);
        getline(file, line);
        getline(file, line);
        getline(file, line);
        getline(file, line);
        getline(file, line);
        getline(file, line);

        while (std::getline(file, line)) {
            if (line[0] == 'E')
                break;

            stringstream sin(line);
            int id;
            double x, y;
            sin >> id;
            sin >> x >> y;

            temp.push_back({ static_cast<int>(x), static_cast<int>(y) });
        }

        unsigned int n = temp.size();

        //calculate distance matrix
        distance_matrix = ComputeEuclideanDistanceMatrix(temp, n);


        file.close();

        return distance_matrix;
    }
    else
    {
        cout << fname << " file not open" << endl;
        return {};
    }
}

int main(int argv, char* argc[])
{
    int threads = 8;
    omp_set_num_threads(threads);
    N = distance_matrix.size();


    std::chrono::time_point<std::chrono::system_clock> start, end;

    //starting time
    start = std::chrono::system_clock::now();
    cout << "start...\n";

    TSP();

    //end time
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
   
    cout << "Minimum cost : " << final_res << '\n';
    cout << ("Path Taken : ");
    for (int i = 0; i < N; i++) cout << final_path[i] << " -> ";
    cout << final_path[N] << '\n';

    cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    cin.get();
    return 0;
}