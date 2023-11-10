#include <iostream>
#include <vector>
#include <random>
using namespace std;

void printMatrix(vector<vector<int>> matrix){
    for(int i = 0; i < matrix.size(); i++){
        for(int j = 0; j < matrix[0].size(); j++){
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

vector<vector<vector<int>>> sumMatrices(const vector<vector<vector<int>>>& matrices) {

    vector<vector<vector<int>>> result(matrices[0].size(), vector<vector<int>>(matrices[0][0].size(), vector<int>(matrices[0][0].size(), 0)));

    for (const auto& matrix : matrices) {
        for (size_t i = 0; i < matrix.size(); ++i) {
            for (size_t j = 0; j < matrix[i].size(); ++j) {
                for (size_t k = 0; k < matrix[i].size(); ++k) {
                    result[i][j][k] += matrix[i][j];
                }
            }
        }
    }
    return result;
}

class Perceptron{
    private:
        double learningRate;
        vector<double> pesos;
        double bias;
        int nInteraciones;

    public:
        Perceptron(double _learningRate = 0.01){
            learningRate = _learningRate;
            pesos = vector<double>(30, 0);
            bias = 0;
            nInteraciones = 200;
        }

        int FuncionActivacion(double x){
            if(x >= 0){
                return 1;
            }
            else{
                return 0;
            }
        }

        void learn(vector<vector<double>> x, vector<double> d){
            int y = -1;

            for (int i = 0; i < nInteraciones; i++){
                for(int j = 0; j < x.size(); j++){
                    if(y == d[j]){
                        break;
                    }

                    double outNumber = 0;
                    for(int k = 0; k < x[j].size(); ++k){
                        outNumber += x[j][k] * pesos[k];
                    }
                    outNumber += bias;

                    y = FuncionActivacion(outNumber);

                    for(int k = 0; k < x[j].size(); k++){
                        pesos[k] = pesos[k] + learningRate * (d[j] - y) * x[j][k];
                    }
                    bias = bias + learningRate * (d[j] - y);
                }
            }
        }

        int prediccion(vector<double> x){
            double outNumber = 0;
            for(int i = 0; i < x.size(); i++){
                outNumber += x[i] * pesos[i];
            }
            outNumber += bias;
            return FuncionActivacion(outNumber);
        }
};

int main(){
    
    // --> Creamos matriz de 5x6
    vector<vector<int>> ej_01(5, vector<int>(6));
    vector<vector<int>> ej_02(5, vector<int>(6));
    vector<vector<int>> ej_03(5, vector<int>(6));
    vector<vector<int>> ej_04(5, vector<int>(6));
    vector<vector<int>> ej_05(5, vector<int>(6));
    vector<vector<int>> ej_06(5, vector<int>(6));

    // --> Inicializamos la matriz con valores aleatorios entre 0 y 1
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1);

    for(int i = 0; i < 5; i++){
        for(int j = 0; j < 6; j++){
            ej_01[i][j] = dis(gen);
            ej_02[i][j] = dis(gen);
            ej_03[i][j] = dis(gen);
            ej_04[i][j] = dis(gen);
            ej_05[i][j] = dis(gen);
            ej_06[i][j] = dis(gen);
        }
    }

    // --> Imprimimos la matriz
    cout << "Ejemplo 01" << endl; printMatrix(ej_01); cout << endl;
    cout << "Ejemplo 02" << endl; printMatrix(ej_02); cout << endl;
    cout << "Ejemplo 03" << endl; printMatrix(ej_03); cout << endl;
    cout << "Ejemplo 04" << endl; printMatrix(ej_04); cout << endl;
    cout << "Ejemplo 05" << endl; printMatrix(ej_05); cout << endl;
    cout << "Ejemplo 06" << endl; printMatrix(ej_06); cout << endl;

    // ################################# MODELOS #################################
    // * Creamos los numeros del 0 al 9 usando matrices de 5x5 usando 1 y 0
    // --> Modelo 0
    vector<vector<int>> modelo0_0 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };
    vector<vector<int>> modelo0_1 = {
        {0, 1, 1, 1, 0},
        {0, 1, 0, 1, 0},
        {0, 1, 0, 1, 0},
        {0, 1, 0, 1, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 0, 0, 0}
    };

    vector<vector<int>> modelo0_2 = {
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 0},
        {0, 1, 0, 1, 0},
        {0, 1, 0, 1, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 0, 0, 0}
    };

    vector<vector<int>> modelo0_3 = {
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 0},
        {0, 1, 0, 1, 0},
        {0, 1, 0, 1, 0},
        {0, 1, 0, 1, 0},
        {0, 1, 1, 1, 0}
    };

    vector<vector<int>> modelo0_4 = {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 1, 1},
        {0, 0, 1, 0, 1},
        {0, 0, 1, 0, 1},
        {0, 0, 1, 0, 1},
        {0, 0, 1, 1, 1}
    };

    vector<vector<int>> modelo0_5 = {
        {0, 0, 0, 0, 0},
        {1, 1, 1, 0, 0},
        {1, 0, 1, 0, 0},
        {1, 0, 1, 0, 0},
        {1, 0, 1, 0, 0},
        {1, 1, 1, 0, 0}
    };

    // --> Modelo 1
    vector<vector<int>> modelo1_0 = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1}
    };

    vector<vector<int>> modelo1_1 = {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 1, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0}
    };

    vector<vector<int>> modelo1_2 = {
        {0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 1, 0, 0, 0}
    };

    vector<vector<int>> modelo1_3 = {
        {0, 0, 1, 0, 0},
        {0, 1, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0}
    };

    vector<vector<int>> modelo1_4 = {
        {0, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    vector<vector<int>> modelo1_5 = {
        {0, 0, 0, 1, 0},
        {0, 0, 1, 1, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0}
    };

    vector<vector<int>> modelo1_6 = {
        {0, 0, 0, 0, 1},
        {0, 0, 0, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0} 
    };

    // --> Modelo 2
    vector<std::vector<int>> modelo2_0 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}
    };

    vector<std::vector<int>> modelo2_1 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}
    };

    vector<std::vector<int>> modelo2_2 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0}
    };

    vector<std::vector<int>> modelo2_3 = {
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}
    };

    // --> Modelo 3


    std::vector<std::vector<int>> modelo3_1 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo3_2 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo3_3 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0}
    };

    std::vector<std::vector<int>> modelo3_4 = {
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo3_5 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo3_6 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo3_7 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0}
    };

    std::vector<std::vector<int>> modelo3_8 = {
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    // --> Modelo 4

    std::vector<std::vector<int>> modelo4_0 = {
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1}
    };

    std::vector<std::vector<int>> modelo4_1 = {
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1}
    };

    std::vector<std::vector<int>> modelo4_2 = {
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0}
    };

    std::vector<std::vector<int>> modelo4_3 = {
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1}
    };

    std::vector<std::vector<int>> modelo4_4 = {
        {0, 1, 0, 0, 1},
        {0, 1, 0, 0, 1},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 0}
    };

    std::vector<std::vector<int>> modelo4_5 = {
        {1, 0, 0, 1, 0},
        {1, 0, 0, 1, 0},
        {1, 1, 1, 1, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 0, 0}
    };

    std::vector<std::vector<int>> modelo4_6 = {
        {0, 0, 0, 0, 0},
        {1, 0, 0, 1, 0},
        {1, 0, 0, 1, 0},
        {1, 1, 1, 1, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 1, 0}
    };

    std::vector<std::vector<int>> modelo4_7 = {
        {0, 0, 0, 0, 0},
        {0, 1, 0, 0, 1},
        {0, 1, 0, 0, 1},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1}
    };


    // --> Modelo 5

    std::vector<std::vector<int>> modelo5_1 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo5_2 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo5_3 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 0}
    };

    std::vector<std::vector<int>> modelo5_4 = {
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo5_5 = {
        {0, 1, 1, 1, 1},
        {0, 1, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo5_6 = {
        {0, 1, 1, 1, 1},
        {0, 1, 0, 0, 0},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo5_7 = {
        {0, 1, 1, 1, 1},
        {0, 1, 0, 0, 0},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 0}
    };

    std::vector<std::vector<int>> modelo5_8 = {
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1},
        {0, 1, 0, 0, 0},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo5_9 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    // --> Modelo 6

    std::vector<std::vector<int>> modelo6_1 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo6_2 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo6_3 = {
        {1, 1, 1, 1, 1},
        {1, 1, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {1, 1, 0, 0, 1},
        {1, 1, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo6_4 = {
        {1, 1, 1, 1, 1},
        {1, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {1, 1, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo6_5 = {
        {1, 1, 1, 1, 1},
        {1, 1, 0, 0, 0},
        {1, 1, 0, 0, 0},
        {1, 1, 1, 1, 1},
        {1, 1, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo6_6 = {
        {1, 1, 1, 1, 0},
        {1, 0, 0, 0, 0},
        {1, 0, 0, 0, 0},
        {1, 1, 1, 1, 0},
        {1, 0, 0, 1, 0},
        {1, 1, 1, 1, 0}
    };

    // --> Modelo 7

    std::vector<std::vector<int>> modelo7_1 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1}
    };

    std::vector<std::vector<int>> modelo7_2 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0}
    };

    std::vector<std::vector<int>> modelo7_3 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 1, 0},
        {0, 0, 0, 1, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0}
    };

    std::vector<std::vector<int>> modelo7_4 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 1, 0},
        {0, 0, 1, 0, 0},
        {0, 1, 0, 0, 0},
        {1, 0, 0, 0, 0}
    };

    std::vector<std::vector<int>> modelo7_5 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 1, 0},
        {1, 1, 1, 1, 1},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0}
    };

    std::vector<std::vector<int>> modelo7_6 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 1, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0}
    };

    std::vector<std::vector<int>> modelo7_7 = {
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 1, 0, 0},
        {0, 1, 0, 0, 0},
        {1, 0, 0, 0, 0}
    };

    // --> Modelo 8

    std::vector<std::vector<int>> modelo8_1 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo8_2 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo8_3 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo8_4 = {
        {0, 1, 1, 1, 0},
        {1, 0, 0, 0, 1},
        {0, 1, 1, 1, 0},
        {0, 1, 1, 1, 0},
        {1, 0, 0, 0, 1},
        {0, 1, 1, 1, 0}
    };

    std::vector<std::vector<int>> modelo8_5 = {
        {0, 1, 1, 1, 0},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {0, 1, 1, 1, 0},
        {1, 0, 0, 0, 1},
        {0, 1, 1, 1, 0}
    };

    std::vector<std::vector<int>> modelo8_6 = {
        {0, 1, 1, 1, 0},
        {1, 0, 0, 0, 1},
        {0, 1, 1, 1, 0},
        {1, 0, 0, 0, 1},
        {1, 0, 0, 0, 1},
        {0, 1, 1, 1, 0}
    };

    std::vector<std::vector<int>> modelo8_7 = {
        {0, 0, 0, 0, 0},
        {1, 1, 1, 1, 0},
        {1, 0, 0, 1, 0},
        {1, 1, 1, 1, 0},
        {1, 0, 0, 1, 0},
        {1, 1, 1, 1, 0}
    };

    std::vector<std::vector<int>> modelo8_8 = {
        {0, 0, 0, 0, 0},
        {0, 1, 1, 1, 1},
        {0, 1, 0, 0, 1},
        {0, 1, 1, 1, 1},
        {0, 1, 0, 0, 1},
        {0, 1, 1, 1, 1}
    };

    // --> Modelo 9

    std::vector<std::vector<int>> modelo9_1 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1}
    };

    std::vector<std::vector<int>> modelo9_2 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo9_3 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 1, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo9_4 = {
        {1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1},
        {1, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 1, 1, 1}
    };

    std::vector<std::vector<int>> modelo9_5 = {
        {0, 1, 1, 1, 0},
        {1, 0, 0, 0, 1},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1}
    };

    std::vector<std::vector<int>> modelo9_6 = {
        {0, 1, 1, 1, 0},
        {1, 0, 0, 0, 1},
        {0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1},
        {0, 0, 0, 0, 1},
        {1, 1, 1, 1, 0}
    };


    // ################################# MAIN #################################
    // --> Creamos el perceptron para cada modelo
    Perceptron p0(0.1);
    Perceptron p1(0.1);
    Perceptron p2(0.1);
    Perceptron p3(0.1);
    Perceptron p4(0.1);
    Perceptron p5(0.1);
    Perceptron p6(0.1);
    Perceptron p7(0.1);
    Perceptron p8(0.1);
    Perceptron p9(0.1);

    // --> Creamos los vectores de entrenamiento (vectores de vectores de vectores)
    vector<vector<vector<int>>> x0 = {
        modelo0_0, modelo0_1, modelo0_2, modelo0_3, modelo0_4, modelo0_5
    };
    vector<vector<vector<int>>> x1 = {
        modelo1_0, modelo1_1, modelo1_2, modelo1_3, modelo1_4, modelo1_5, modelo1_6
    };
    vector<vector<vector<int>>> x2 = {
        modelo2_0, modelo2_1, modelo2_2, modelo2_3
    };
    vector<vector<vector<int>>> x3 = {
        modelo3_1, modelo3_2, modelo3_3, modelo3_4, modelo3_5, modelo3_6, modelo3_7, modelo3_8 
    };
    vector<vector<vector<int>>> x4 = {
        modelo4_0, modelo4_1, modelo4_2, modelo4_3, modelo4_4, modelo4_5, modelo4_6, modelo4_7
    };
    vector<vector<vector<int>>> x5 = {
        modelo5_1, modelo5_2, modelo5_3, modelo5_4, modelo5_5, modelo5_6, modelo5_7, modelo5_8, modelo5_9
    };
    vector<vector<vector<int>>> x6 = {
        modelo6_1, modelo6_2, modelo6_3, modelo6_4, modelo6_5, modelo6_6
    };
    vector<vector<vector<int>>> x7 = {
        modelo7_1, modelo7_2, modelo7_3, modelo7_4, modelo7_5, modelo7_6, modelo7_7
    };
    vector<vector<vector<int>>> x8 = {
        modelo8_1, modelo8_2, modelo8_3, modelo8_4, modelo8_5, modelo8_6, modelo8_7, modelo8_8
    };
    vector<vector<vector<int>>> x9 = {
        modelo9_1, modelo9_2, modelo9_3, modelo9_4, modelo9_5, modelo9_6
    };

    // --> Creamos los vectores de salida
    vector<double> d0(x0.size(), 1);
    d0.insert(d0.end(), x1.size(), 0);
    d0.insert(d0.end(), x2.size(), 0);
    d0.insert(d0.end(), x3.size(), 0);
    d0.insert(d0.end(), x4.size(), 0);
    d0.insert(d0.end(), x5.size(), 0);
    d0.insert(d0.end(), x6.size(), 0);
    d0.insert(d0.end(), x7.size(), 0);
    d0.insert(d0.end(), x8.size(), 0);
    d0.insert(d0.end(), x9.size(), 0);

    vector<double> d1(x0.size(), 0);
    d1.insert(d1.end(), x1.size(), 1);
    d1.insert(d1.end(), x2.size(), 0);
    d1.insert(d1.end(), x3.size(), 0);
    d1.insert(d1.end(), x4.size(), 0);
    d1.insert(d1.end(), x5.size(), 0);
    d1.insert(d1.end(), x6.size(), 0);
    d1.insert(d1.end(), x7.size(), 0);
    d1.insert(d1.end(), x8.size(), 0);
    d1.insert(d1.end(), x9.size(), 0);

    vector<double> d2(x0.size(), 0);
    d2.insert(d2.end(), x1.size(), 0);
    d2.insert(d2.end(), x2.size(), 1);
    d2.insert(d2.end(), x3.size(), 0);
    d2.insert(d2.end(), x4.size(), 0);
    d2.insert(d2.end(), x5.size(), 0);
    d2.insert(d2.end(), x6.size(), 0);
    d2.insert(d2.end(), x7.size(), 0);
    d2.insert(d2.end(), x8.size(), 0);
    d2.insert(d2.end(), x9.size(), 0);

    vector<double> d3(x0.size(), 0);
    d3.insert(d3.end(), x1.size(), 0);
    d3.insert(d3.end(), x2.size(), 0);
    d3.insert(d3.end(), x3.size(), 1);
    d3.insert(d3.end(), x4.size(), 0);
    d3.insert(d3.end(), x5.size(), 0);
    d3.insert(d3.end(), x6.size(), 0);
    d3.insert(d3.end(), x7.size(), 0);
    d3.insert(d3.end(), x8.size(), 0);
    d3.insert(d3.end(), x9.size(), 0);

    vector<double> d4(x0.size(), 0);
    d4.insert(d4.end(), x1.size(), 0);
    d4.insert(d4.end(), x2.size(), 0);
    d4.insert(d4.end(), x3.size(), 0);
    d4.insert(d4.end(), x4.size(), 1);
    d4.insert(d4.end(), x5.size(), 0);
    d4.insert(d4.end(), x6.size(), 0);
    d4.insert(d4.end(), x7.size(), 0);
    d4.insert(d4.end(), x8.size(), 0);
    d4.insert(d4.end(), x9.size(), 0);

    vector<double> d5(x0.size(), 0);
    d5.insert(d5.end(), x1.size(), 0);
    d5.insert(d5.end(), x2.size(), 0);
    d5.insert(d5.end(), x3.size(), 0);
    d5.insert(d5.end(), x4.size(), 0);
    d5.insert(d5.end(), x5.size(), 1);
    d5.insert(d5.end(), x6.size(), 0);
    d5.insert(d5.end(), x7.size(), 0);
    d5.insert(d5.end(), x8.size(), 0);
    d5.insert(d5.end(), x9.size(), 0);

    vector<double> d6(x0.size(), 0);
    d6.insert(d6.end(), x1.size(), 0);
    d6.insert(d6.end(), x2.size(), 0);
    d6.insert(d6.end(), x3.size(), 0);
    d6.insert(d6.end(), x4.size(), 0);
    d6.insert(d6.end(), x5.size(), 0);
    d6.insert(d6.end(), x6.size(), 1);
    d6.insert(d6.end(), x7.size(), 0);
    d6.insert(d6.end(), x8.size(), 0);
    d6.insert(d6.end(), x9.size(), 0);

    vector<double> d7(x0.size(), 0);
    d7.insert(d7.end(), x1.size(), 0);
    d7.insert(d7.end(), x2.size(), 0);
    d7.insert(d7.end(), x3.size(), 0);
    d7.insert(d7.end(), x4.size(), 0);
    d7.insert(d7.end(), x5.size(), 0);
    d7.insert(d7.end(), x6.size(), 0);
    d7.insert(d7.end(), x7.size(), 1);
    d7.insert(d7.end(), x8.size(), 0);
    d7.insert(d7.end(), x9.size(), 0);

    vector<double> d8(x0.size(), 0);
    d8.insert(d8.end(), x1.size(), 0);
    d8.insert(d8.end(), x2.size(), 0);
    d8.insert(d8.end(), x3.size(), 0);
    d8.insert(d8.end(), x4.size(), 0);
    d8.insert(d8.end(), x5.size(), 0);
    d8.insert(d8.end(), x6.size(), 0);
    d8.insert(d8.end(), x7.size(), 0);
    d8.insert(d8.end(), x8.size(), 1);
    d8.insert(d8.end(), x9.size(), 0);

    vector<double> d9(x0.size(), 0);
    d9.insert(d9.end(), x1.size(), 0);
    d9.insert(d9.end(), x2.size(), 0);
    d9.insert(d9.end(), x3.size(), 0);
    d9.insert(d9.end(), x4.size(), 0);
    d9.insert(d9.end(), x5.size(), 0);
    d9.insert(d9.end(), x6.size(), 0);
    d9.insert(d9.end(), x7.size(), 0);
    d9.insert(d9.end(), x8.size(), 0);
    d9.insert(d9.end(), x9.size(), 1);


    // --> Entrenamos los perceptrones

    p0.learn(x,d0);
    p1.learn(x,d1);
    p2.learn(,d2);
    p3.learn(,d3);
    p4.learn(,d4);
    p5.learn(,d5);
    p6.learn(,d6);
    p7.learn(,d7);
    p8.learn(,d8);
    p9.learn(,d9);


    
    // std::vector<std::vector<std::vector<int>>> matrix1 = {
    //     {{1, 2, 3}, {4, 5, 6}},
    //     {{7, 8, 9}, {10, 11, 12}}
    // };

    // std::vector<std::vector<std::vector<int>>> matrix2 = {
    //     {{2, 2, 2}, {2, 2, 2}},
    //     {{3, 3, 3}, {3, 3, 3}}
    // };

    // // Sumar las matrices utilizando la función sumMatrices
    // std::vector<std::vector<std::vector<int>>> result = sumMatrices(matrix2);

    // p1.learn(d0);

    // Mostrar el resultado
    // for (const auto& slice : result) {
    //     for (const auto& row : slice) {
    //         for (int value : row) {
    //             std::cout << value << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}