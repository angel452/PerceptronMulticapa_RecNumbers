#include <iostream>
#include <vector>
#include <random>
using namespace std;

#include "data_entranamiento.hpp"
#include "data_test.hpp"

void printMatrix(vector<vector<int>> matrix){
    for(int i = 0; i < matrix.size(); i++){
        for(int j = 0; j < matrix[0].size(); j++){
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
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
            bias = 0.0;
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

        void learn(vector<vector<double>>& x, vector<int>& d){
            pesos = vector<double>(30, 0.0);
            int y = -1;

            // --> Iteramos b veces para obtener los pesos
            for (int i = 0; i < nInteraciones; ++i){
                for(int j = 0; j < x.size(); ++j){
                    //std::cout << "y: " << y << std::endl;
                    //if(y == d[j]){
                    //    cout << "Saimos del learn" << endl;
                    //    break;
                    //}

                    double outNumber = std::inner_product(x[j].begin(), x[j].end(), pesos.begin(), 0.0) + bias;
                    y = FuncionActivacion(outNumber);
                    
                    /*double outNumber = 0.0;
                    for(int k = 0; k < x[j].size(); ++k){
                        outNumber += x[j][k] * pesos[k];
                    }
                    outNumber += bias;
                    //std::cout << "outnumber: " << outNumber << std::endl;
                    y = FuncionActivacion(outNumber);*/

                    for(int k = 0; k < x[j].size(); ++k){
                    //for(int k = 0; k < pesos.size(); ++k){
                        pesos[k] += learningRate * x[j][k] * (d[j] - y);
                    }
                    // bias = bias + learningRate * (d[j] - y);
                }
            }
        }

        int predict(vector<double> x){

            double outNumber = std::inner_product(x.begin(), x.end(), pesos.begin(), 0.0) + bias;
            /*
            double outNumber = 0;
            for(int i = 0; i < x.size(); i++){
                outNumber += x[i] * pesos[i];
            }
            outNumber += bias;*/
            return FuncionActivacion(outNumber);
        }
};

int main(){
    // ################################# MAIN #################################
    // // --> Creamos el perceptron para cada modelo
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
    // x0 = [ex0_0, ex0_01...]
    vector<vector<double>> x0 = {
        modelo0_0, modelo0_1, modelo0_2, modelo0_3, modelo0_4, modelo0_5
    };

    /*
    cout << "--------------------------------------" << endl;
    cout << "TEST0: " << x0.size() << endl;
    for(int i = 0; i < x0.size(); i++){
        for(int j = 0; j < x0[0].size(); j++){
            cout << x0[i][j] << " ";
        }
        cout << endl;
    }
    cout << "--------------------------------------" << endl;
    */

    vector<vector<double>> x1 = {
        modelo1_0, modelo1_1, modelo1_2, modelo1_3, modelo1_4, modelo1_5, modelo1_6
    };
    vector<vector<double>> x2 = {
        modelo2_0, modelo2_1, modelo2_2, modelo2_3
    };
    vector<vector<double>> x3 = {
        modelo3_1, modelo3_2, modelo3_3, modelo3_4, modelo3_5, modelo3_6, modelo3_7, modelo3_8 
    };
    vector<vector<double>> x4 = {
        modelo4_0, modelo4_1, modelo4_2, modelo4_3, modelo4_4, modelo4_5, modelo4_6, modelo4_7
    };
    vector<vector<double>> x5 = {
        modelo5_1, modelo5_2, modelo5_3, modelo5_4, modelo5_5, modelo5_6, modelo5_7, modelo5_8, modelo5_9
    };
    vector<vector<double>> x6 = {
        modelo6_1, modelo6_2, modelo6_3, modelo6_4, modelo6_5, modelo6_6
    };
    vector<vector<double>> x7 = {
        modelo7_1, modelo7_2, modelo7_3, modelo7_4, modelo7_5, modelo7_6, modelo7_7
    };
    vector<vector<double>> x8 = {
        modelo8_1, modelo8_2, modelo8_3, modelo8_4, modelo8_5, modelo8_6, modelo8_7, modelo8_8
    };
    vector<vector<double>> x9 = {
        modelo9_1, modelo9_2, modelo9_3, modelo9_4, modelo9_5, modelo9_6
    };

    // --> Creamos los vectores de salida
    vector<int> d0(x0.size(), 1);
    d0.insert(d0.end(), x1.size(), 0);
    d0.insert(d0.end(), x2.size(), 0);
    d0.insert(d0.end(), x3.size(), 0);
    d0.insert(d0.end(), x4.size(), 0);
    d0.insert(d0.end(), x5.size(), 0);
    d0.insert(d0.end(), x6.size(), 0);
    d0.insert(d0.end(), x7.size(), 0);
    d0.insert(d0.end(), x8.size(), 0);
    d0.insert(d0.end(), x9.size(), 0);

    /*
    cout << "--------------------------------------" << endl;
    cout << "TEST0: " << d0.size() << endl;
    for(int i = 0; i < d0.size(); i++){
        cout << d0[i] << " ";
    }
    cout << endl;
    cout << "--------------------------------------" << endl;
    */

    vector<int> d1(x0.size(), 0);
    d1.insert(d1.end(), x1.size(), 1);
    d1.insert(d1.end(), x2.size(), 0);
    d1.insert(d1.end(), x3.size(), 0);
    d1.insert(d1.end(), x4.size(), 0);
    d1.insert(d1.end(), x5.size(), 0);
    d1.insert(d1.end(), x6.size(), 0);
    d1.insert(d1.end(), x7.size(), 0);
    d1.insert(d1.end(), x8.size(), 0);
    d1.insert(d1.end(), x9.size(), 0);

    vector<int> d2(x0.size(), 0);
    d2.insert(d2.end(), x1.size(), 0);
    d2.insert(d2.end(), x2.size(), 1);
    d2.insert(d2.end(), x3.size(), 0);
    d2.insert(d2.end(), x4.size(), 0);
    d2.insert(d2.end(), x5.size(), 0);
    d2.insert(d2.end(), x6.size(), 0);
    d2.insert(d2.end(), x7.size(), 0);
    d2.insert(d2.end(), x8.size(), 0);
    d2.insert(d2.end(), x9.size(), 0);

    vector<int> d3(x0.size(), 0);
    d3.insert(d3.end(), x1.size(), 0);
    d3.insert(d3.end(), x2.size(), 0);
    d3.insert(d3.end(), x3.size(), 1);
    d3.insert(d3.end(), x4.size(), 0);
    d3.insert(d3.end(), x5.size(), 0);
    d3.insert(d3.end(), x6.size(), 0);
    d3.insert(d3.end(), x7.size(), 0);
    d3.insert(d3.end(), x8.size(), 0);
    d3.insert(d3.end(), x9.size(), 0);

    vector<int> d4(x0.size(), 0);
    d4.insert(d4.end(), x1.size(), 0);
    d4.insert(d4.end(), x2.size(), 0);
    d4.insert(d4.end(), x3.size(), 0);
    d4.insert(d4.end(), x4.size(), 1);
    d4.insert(d4.end(), x5.size(), 0);
    d4.insert(d4.end(), x6.size(), 0);
    d4.insert(d4.end(), x7.size(), 0);
    d4.insert(d4.end(), x8.size(), 0);
    d4.insert(d4.end(), x9.size(), 0);

    vector<int> d5(x0.size(), 0);
    d5.insert(d5.end(), x1.size(), 0);
    d5.insert(d5.end(), x2.size(), 0);
    d5.insert(d5.end(), x3.size(), 0);
    d5.insert(d5.end(), x4.size(), 0);
    d5.insert(d5.end(), x5.size(), 1);
    d5.insert(d5.end(), x6.size(), 0);
    d5.insert(d5.end(), x7.size(), 0);
    d5.insert(d5.end(), x8.size(), 0);
    d5.insert(d5.end(), x9.size(), 0);

    vector<int> d6(x0.size(), 0);
    d6.insert(d6.end(), x1.size(), 0);
    d6.insert(d6.end(), x2.size(), 0);
    d6.insert(d6.end(), x3.size(), 0);
    d6.insert(d6.end(), x4.size(), 0);
    d6.insert(d6.end(), x5.size(), 0);
    d6.insert(d6.end(), x6.size(), 1);
    d6.insert(d6.end(), x7.size(), 0);
    d6.insert(d6.end(), x8.size(), 0);
    d6.insert(d6.end(), x9.size(), 0);

    vector<int> d7(x0.size(), 0);
    d7.insert(d7.end(), x1.size(), 0);
    d7.insert(d7.end(), x2.size(), 0);
    d7.insert(d7.end(), x3.size(), 0);
    d7.insert(d7.end(), x4.size(), 0);
    d7.insert(d7.end(), x5.size(), 0);
    d7.insert(d7.end(), x6.size(), 0);
    d7.insert(d7.end(), x7.size(), 1);
    d7.insert(d7.end(), x8.size(), 0);
    d7.insert(d7.end(), x9.size(), 0);

    vector<int> d8(x0.size(), 0);
    d8.insert(d8.end(), x1.size(), 0);
    d8.insert(d8.end(), x2.size(), 0);
    d8.insert(d8.end(), x3.size(), 0);
    d8.insert(d8.end(), x4.size(), 0);
    d8.insert(d8.end(), x5.size(), 0);
    d8.insert(d8.end(), x6.size(), 0);
    d8.insert(d8.end(), x7.size(), 0);
    d8.insert(d8.end(), x8.size(), 1);
    d8.insert(d8.end(), x9.size(), 0);

    vector<int> d9(x0.size(), 0);
    d9.insert(d9.end(), x1.size(), 0);
    d9.insert(d9.end(), x2.size(), 0);
    d9.insert(d9.end(), x3.size(), 0);
    d9.insert(d9.end(), x4.size(), 0);
    d9.insert(d9.end(), x5.size(), 0);
    d9.insert(d9.end(), x6.size(), 0);
    d9.insert(d9.end(), x7.size(), 0);
    d9.insert(d9.end(), x8.size(), 0);
    d9.insert(d9.end(), x9.size(), 1);


    // añadir todos los datos a una sola matriz

    std::vector<std::vector<double>> x;
    x.insert(x.end(), x0.begin(), x0.end());
    x.insert(x.end(), x1.begin(), x1.end());
    x.insert(x.end(), x2.begin(), x2.end());
    x.insert(x.end(), x3.begin(), x3.end());
    x.insert(x.end(), x4.begin(), x4.end());
    x.insert(x.end(), x5.begin(), x5.end());
    x.insert(x.end(), x6.begin(), x6.end());
    x.insert(x.end(), x7.begin(), x7.end());
    x.insert(x.end(), x8.begin(), x8.end());
    x.insert(x.end(), x9.begin(), x9.end());

    // --> Imprimir x
    /*
    std::cout << "x: " << x.size() << std::endl;
    for(int i = 0; i < x.size(); i++){
        for(int j = 0; j < x[0].size(); j++){
            cout << x[i][j] << " ";
        }
        cout << endl;
    }
    */

    // std::cout << "d0: " << d0.size() << std::endl;

    // for(int i = 0; i < d0.size(); i++)
    // {
    //     std::cout <<  d0[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "d9: " << d9.size() << std::endl;

    // for(int i = 0; i < d8.size(); i++)
    // {
    //     std::cout <<  d8[i] << " ";
    // }
    // std::cout << std::endl;


    // std::cout << "d0: " << d1.size() << std::endl;


    // // --> Entrenamos los perceptrones

    std::cout << "learn0\n";
    p0.learn(x,d0);
    std::cout << "learn1\n";
    p1.learn(x,d1);
    std::cout << "learn2\n";
    p2.learn(x,d2);
    std::cout << "learn3\n";
    p3.learn(x,d3);
    std::cout << "learn4\n";
    p4.learn(x,d4);
    std::cout << "learn5\n";
    p5.learn(x,d5);
    std::cout << "learn6\n";
    p6.learn(x,d6);
    std::cout << "learn7\n";
    p7.learn(x,d7);
    std::cout << "learn8\n";
    p8.learn(x,d8);
    std::cout << "learn9\n";
    p9.learn(x,d9);

    std::cout << "el test cero es un:\n" ;
    std::cout << "Numero 0:" << p0.predict(test0) << std::endl;
    std::cout << "Numero 1:" << p1.predict(test0) << std::endl;
    std::cout << "Numero 2:" << p2.predict(test0) << std::endl;
    std::cout << "Numero 3:" << p3.predict(test0) << std::endl;
    std::cout << "Numero 4:" << p4.predict(test0) << std::endl;
    std::cout << "Numero 5:" << p5.predict(test0) << std::endl;
    std::cout << "Numero 6:" << p6.predict(test0) << std::endl;
    std::cout << "Numero 7:" << p7.predict(test0) << std::endl;
    std::cout << "Numero 8:" << p8.predict(test0) << std::endl;
    std::cout << "Numero 9:" << p9.predict(test0) << std::endl;

    std::cout << "\nel test uno es un:\n" ;
    std::cout << "Numero 0:" << p0.predict(test1)<<std::endl;
    std::cout << "Numero 1:" << p1.predict(test1)<<std::endl;
    std::cout << "Numero 2:" << p2.predict(test1)<<std::endl;
    std::cout << "Numero 3:" << p3.predict(test1)<<std::endl;
    std::cout << "Numero 4:" << p4.predict(test1)<<std::endl;
    std::cout << "Numero 5:" << p5.predict(test1)<<std::endl;
    std::cout << "Numero 6:" << p6.predict(test1)<<std::endl;
    std::cout << "Numero 7:" << p7.predict(test1)<<std::endl;
    std::cout << "Numero 8:" << p8.predict(test1)<<std::endl;
    std::cout << "Numero 9:" << p9.predict(test1)<<std::endl;

    std::cout << "\nel test dos es un:\n";
    std::cout << "Numero 0:" << p0.predict(test2)<<std::endl;
    std::cout << "Numero 1:" << p1.predict(test2)<<std::endl;
    std::cout << "Numero 2:" << p2.predict(test2)<<std::endl;
    std::cout << "Numero 3:" << p3.predict(test2)<<std::endl;
    std::cout << "Numero 4:" << p4.predict(test2)<<std::endl;
    std::cout << "Numero 5:" << p5.predict(test2)<<std::endl;
    std::cout << "Numero 6:" << p6.predict(test2)<<std::endl;
    std::cout << "Numero 7:" << p7.predict(test2)<<std::endl;
    std::cout << "Numero 8:" << p8.predict(test2)<<std::endl;
    std::cout << "Numero 9:" << p9.predict(test2)<<std::endl;


    std::cout << "\nel test tres es un:\n";
    std::cout << "Numero 0:" << p0.predict(test3)<<std::endl;
    std::cout << "Numero 1:" << p1.predict(test3)<<std::endl;
    std::cout << "Numero 2:" << p2.predict(test3)<<std::endl;
    std::cout << "Numero 3:" << p3.predict(test3)<<std::endl;
    std::cout << "Numero 4:" << p4.predict(test3)<<std::endl;
    std::cout << "Numero 5:" << p5.predict(test3)<<std::endl;
    std::cout << "Numero 6:" << p6.predict(test3)<<std::endl;
    std::cout << "Numero 7:" << p7.predict(test3)<<std::endl;
    std::cout << "Numero 8:" << p8.predict(test3)<<std::endl;
    std::cout << "Numero 9:" << p9.predict(test3)<<std::endl;

    std::cout << "\nel test cuatro es un:\n";
    std::cout << "Numero 0:" << p0.predict(test4)<<std::endl;
    std::cout << "Numero 1:" << p1.predict(test4)<<std::endl;
    std::cout << "Numero 2:" << p2.predict(test4)<<std::endl;
    std::cout << "Numero 3:" << p3.predict(test4)<<std::endl;
    std::cout << "Numero 4:" << p4.predict(test4)<<std::endl;
    std::cout << "Numero 5:" << p5.predict(test4)<<std::endl;
    std::cout << "Numero 6:" << p6.predict(test4)<<std::endl;
    std::cout << "Numero 7:" << p7.predict(test4)<<std::endl;
    std::cout << "Numero 8:" << p8.predict(test4)<<std::endl;
    std::cout << "Numero 9:" << p9.predict(test4)<<std::endl;

    std::cout << "\nel test cinco es un:\n";
    std::cout << "Numero 0:" << p0.predict(test5)<<std::endl;
    std::cout << "Numero 1:" << p1.predict(test5)<<std::endl;
    std::cout << "Numero 2:" << p2.predict(test5)<<std::endl;
    std::cout << "Numero 3:" << p3.predict(test5)<<std::endl;
    std::cout << "Numero 4:" << p4.predict(test5)<<std::endl;
    std::cout << "Numero 5:" << p5.predict(test5)<<std::endl;
    std::cout << "Numero 6:" << p6.predict(test5)<<std::endl;
    std::cout << "Numero 7:" << p7.predict(test5)<<std::endl;
    std::cout << "Numero 8:" << p8.predict(test5)<<std::endl;
    std::cout << "Numero 9:" << p9.predict(test5)<<std::endl;

    std::cout << "\nel test seis es un:\n";
    std::cout << "Numero 0:" << p0.predict(test6)<<std::endl;
    std::cout << "Numero 1:" << p1.predict(test6)<<std::endl;
    std::cout << "Numero 2:" << p2.predict(test6)<<std::endl;
    std::cout << "Numero 3:" << p3.predict(test6)<<std::endl;
    std::cout << "Numero 4:" << p4.predict(test6)<<std::endl;
    std::cout << "Numero 5:" << p5.predict(test6)<<std::endl;
    std::cout << "Numero 6:" << p6.predict(test6)<<std::endl;
    std::cout << "Numero 7:" << p7.predict(test6)<<std::endl;
    std::cout << "Numero 8:" << p8.predict(test6)<<std::endl;
    std::cout << "Numero 9:" << p9.predict(test6)<<std::endl;

    std::cout << "\nel test siete es un:\n";
    std::cout << "Numero 0:" << p0.predict(test7)<<std::endl;
    std::cout << "Numero 1:" << p1.predict(test7)<<std::endl;
    std::cout << "Numero 2:" << p2.predict(test7)<<std::endl;
    std::cout << "Numero 3:" << p3.predict(test7)<<std::endl;
    std::cout << "Numero 4:" << p4.predict(test7)<<std::endl;
    std::cout << "Numero 5:" << p5.predict(test7)<<std::endl;
    std::cout << "Numero 6:" << p6.predict(test7)<<std::endl;
    std::cout << "Numero 7:" << p7.predict(test7)<<std::endl;
    std::cout << "Numero 8:" << p8.predict(test7)<<std::endl;
    std::cout << "Numero 9:" << p9.predict(test7)<<std::endl;

    std::cout << "\nel test ocho es un:\n";
    std::cout << "Numero 0:" << p0.predict(test8)<<std::endl;
    std::cout << "Numero 1:" << p1.predict(test8)<<std::endl;
    std::cout << "Numero 2:" << p2.predict(test8)<<std::endl;
    std::cout << "Numero 3:" << p3.predict(test8)<<std::endl;
    std::cout << "Numero 4:" << p4.predict(test8)<<std::endl;
    std::cout << "Numero 5:" << p5.predict(test8)<<std::endl;
    std::cout << "Numero 6:" << p6.predict(test8)<<std::endl;
    std::cout << "Numero 7:" << p7.predict(test8)<<std::endl;
    std::cout << "Numero 8:" << p8.predict(test8)<<std::endl;
    std::cout << "Numero 9:" << p9.predict(test8)<<std::endl;

    std::cout << "\nel test nueve es un:\n";
    std::cout << "Numero 0:" << p0.predict(test9)<<std::endl;
    std::cout << "Numero 1:" << p1.predict(test9)<<std::endl;
    std::cout << "Numero 2:" << p2.predict(test9)<<std::endl;
    std::cout << "Numero 3:" << p3.predict(test9)<<std::endl;
    std::cout << "Numero 4:" << p4.predict(test9)<<std::endl;
    std::cout << "Numero 5:" << p5.predict(test9)<<std::endl;
    std::cout << "Numero 6:" << p6.predict(test9)<<std::endl;
    std::cout << "Numero 7:" << p7.predict(test9)<<std::endl;
    std::cout << "Numero 8:" << p8.predict(test9)<<std::endl;
    std::cout << "Numero 9:" << p9.predict(test9)<<std::endl;


    return 0;
}