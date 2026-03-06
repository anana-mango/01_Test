using System;
using System.Numerics;
using static System.Math;

namespace ParameterReviewer
{
    class FFT
    {
        double PI = 3.1415926535897;
        static int N;        // FFT 주파수 수 (2^n 만 가능)
        static int lgN;      // N의 log2 값
        bool Inverse;        // inverse FFT >> true
        Complex[] output;    // FFT 결과 complex 배열
        Complex[] omega;     // FFT시 사용하는 omega 배열

        // 그냥 쓰는 친구들. 선언을 미리 해놓는게 빠른 것 같아 해놓는다. 
        int[] int1;


        // Class Initialize
        public FFT(int n, bool inverse)
        {
            N = n;
            // lgN 계산
            int tmN = 0;
            for (int i = n; i > 1; i >>= 1) tmN++;
            lgN = tmN;
            Inverse = inverse;

            // 배열 초기 선언
            output = new Complex[N];
            omega = new Complex[lgN];

        }
        /// <summary>
        /// /=============    Public Functions    =================
        /// </summary>
        // Main Doing FFT function
        public Complex[] DoFFT(int[] inarr, int FilSize)
        {
            // bitPreProcess 에서 사용할 int1 배열 512size 0 채워서 초기화
            int1 = new int[N];
            
            // make Symmetry & index bit conversion
            bitPreProcess(inarr, FilSize);
            // make omega
            MakeOmega();

            int m = 1;
            for (int s = 0; s < lgN; ++s)   // 9 = lgN
            {
                m <<= 1; // m * 2 = region, 2부터 시작
                for (int k = 0; k < N; k += m)  // region 씩 건너뜀
                {
                    Complex current_omega = 1;
                    for (int j = 0; j < (m >> 1); ++j)  // region m / 2, 즉 region의 반만큼 
                    {
                        Complex t = current_omega * output[k + j + (m >> 1)];  // y = p + w*q 에서 Q, omega 곱함
                        Complex u = output[k + j];     // 그 때 P, 이번 region에서 j번째에 해당함
                        output[k + j] = u + t;   // 위 식에서 y. 전체 region중 plus conjugate를 가지는 부분
                        output[k + j + (m >> 1)] = u - t;   // 위 식에서 y. 전체 식 중 - conjugate를 가지는 부분.
                        current_omega *= omega[s];   // omega 맞춰서 거듭제곱    
                    }
                }
            }
            if (Inverse == true)
                for (int i = 0; i < N; ++i)
                    output[i] /= N;
            return output;
        }
        // turn Complex to Magnitude & make log scale([dB]) function
        // also correct deviation due to bypass magnitude value 84.2883987859147
        public double[] DoMagnitude(Complex[] freq)
        {
            double[] mag = new double[N];
            for (int i = 0; i < freq.Length; i++)
            {
                mag[i] = 20 * Math.Log10(freq[i].Magnitude) - 84.2883987859147;
            }
            return mag;
        }


        /// <summary>
        /// /=============    Private Functions    =================
        /// </summary>
        // bit Pre Processing
        private void bitPreProcess(int[] inarr, int FilSize)
        {
            // 512 size, make Symmetry filter array out1
            for (int i = 0; i < FilSize - 1; i++)
            {
                int1[i] = inarr[i];
                int1[FilSize * 2 - 2 - i] = inarr[i];
            }
            int1[FilSize - 1] = inarr[FilSize - 1];

            // make index bit conversion
            for (int i = 0; i < N; i++)
            {
                int index = i, rev = 0;
                for (int j = 0; j < lgN; ++j)
                {
                    rev = (rev << 1) | (index & 1);
                    index >>= 1;
                }
                output[rev] = int1[i];
            }
        }
        // Make Omega for FFT - inverse ver / normal ver
        private void MakeOmega()
        {
            int m = 1;

            for (int s = 0; s < lgN; ++s)
            {
                m <<= 1;
                if (Inverse)
                    omega[s] = Complex.Exp(new Complex(0, 2.0 * PI / m));
                else
                    omega[s] = Complex.Exp(new Complex(0, -2.0 * PI / m));
            }
        }

        

    }   
}
