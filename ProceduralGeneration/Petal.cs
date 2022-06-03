using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Assign coordinates to the points of the bicubic patch to generate petals

public class Petal : MonoBehaviour
{
    public GameObject vertexMarkerPrefab;
    public List<GameObject> row0Points;
    public List<GameObject> row1Points;
    public List<GameObject> row2Points;
    public List<GameObject> row3Points;

    public int num_vertices_s;
    public int num_vertices_t;
    public bool showVertices;

    private List<List<GameObject>> pointsGrid;
    private List<List<Vector3>> vertices;
    private List<List<GameObject>> vertexMarkers;

    public void ResetControlPoints()
    {
        for (int i = 0; i < pointsGrid.Count; i++) {
            for (int j = 0; j < pointsGrid[0].Count; j++) {
                pointsGrid[i][j].transform.position = new Vector3((float)i / (pointsGrid.Count - 1), 0, (float)j / (pointsGrid[0].Count - 1));    
            }
        }
    }

    public void PetalControlPoints() 
    {
        for (int i = 0; i < 4; i++) {
            pointsGrid[0][i].transform.position = new Vector3(0, 0, 0.5f);
            pointsGrid[3][i].transform.position = new Vector3(1.0f, 0, 0.5f);
        }

        float upperVertexForward = Random.Range(-0.25f, 0.5f);
        pointsGrid[2][3].transform.position += new Vector3(upperVertexForward, 0, 0);
        pointsGrid[2][0].transform.position += new Vector3(upperVertexForward, 0, 0);
        
        float upperVertexSide = Random.Range(-0.2f, 0.15f);
        pointsGrid[2][3].transform.position += new Vector3(0, 0, upperVertexSide);
        pointsGrid[2][0].transform.position -= new Vector3(0, 0, upperVertexSide);

        float lowerVertexForwardDist = Random.Range(0.1f, 1.0f);
        pointsGrid[1][3].transform.position = pointsGrid[2][3].transform.position;
        pointsGrid[1][0].transform.position = pointsGrid[2][0].transform.position;
        pointsGrid[1][3].transform.position -= new Vector3(lowerVertexForwardDist, 0, 0);
        pointsGrid[1][0].transform.position -= new Vector3(lowerVertexForwardDist, 0, 0);
        pointsGrid[1][3].transform.position = new Vector3(Mathf.Max(pointsGrid[1][3].transform.position.x, 0), pointsGrid[1][3].transform.position.y, pointsGrid[1][3].transform.position.z);
        pointsGrid[1][0].transform.position = new Vector3(Mathf.Max(pointsGrid[1][0].transform.position.x, 0), pointsGrid[1][0].transform.position.y, pointsGrid[1][0].transform.position.z);

        float lowerVertexSide = Random.Range(-0.35f, 0.075f);
        pointsGrid[1][3].transform.position += new Vector3(0, 0, lowerVertexSide);
        pointsGrid[1][0].transform.position -= new Vector3(0, 0, lowerVertexSide);

        pointsGrid[2][2].transform.position = new Vector3(0.666f, 0, 0.5f);
        pointsGrid[2][1].transform.position = new Vector3(0.666f, 0, 0.5f);
        pointsGrid[1][2].transform.position = new Vector3(0.333f, 0, 0.5f);
        pointsGrid[1][1].transform.position = new Vector3(0.333f, 0, 0.5f);

        float outerForwardPitch = Random.Range(-0.5f, 0.5f);
        pointsGrid[2][3].transform.position += new Vector3(0, outerForwardPitch, 0);
        pointsGrid[2][0].transform.position += new Vector3(0, outerForwardPitch, 0);

        float outerBackwardPitch = Random.Range(-0.5f, 0.5f);
        pointsGrid[1][3].transform.position += new Vector3(0, outerBackwardPitch, 0);
        pointsGrid[1][0].transform.position += new Vector3(0, outerBackwardPitch, 0);

        float innerForwardPitchDist = Random.Range(0.05f, 0.25f);
        pointsGrid[2][2].transform.position += new Vector3(0, outerForwardPitch - innerForwardPitchDist, 0);
        pointsGrid[2][1].transform.position += new Vector3(0, outerForwardPitch - innerForwardPitchDist, 0);

        float innerBackwardPitchDist = Random.Range(0.05f, 0.25f);
        pointsGrid[1][2].transform.position += new Vector3(0, outerBackwardPitch - innerBackwardPitchDist, 0);
        pointsGrid[1][1].transform.position += new Vector3(0, outerBackwardPitch - innerBackwardPitchDist, 0);

    }   

    // Start is called before the first frame update
    void Start()
    {   
        // Add markers for each vertex in the mesh

        // vertexMarkers = new List<List<GameObject>>();
        // for (int i = 0; i < num_vertices_s; i++) {
        //     List<GameObject> vertexMarkerRow = new List<GameObject>();
        //     for (int j = 0; j < num_vertices_t; j++) {
        //         GameObject vertexMarker = Instantiate(vertexMarkerPrefab, new Vector3(0, 0, 0), Quaternion.identity);
        //         vertexMarkerRow.Add(vertexMarker);
        //     }
        //     vertexMarkers.Add(vertexMarkerRow);
        // }

        pointsGrid = new List<List<GameObject>>();
        pointsGrid.Add(row0Points);
        pointsGrid.Add(row1Points);
        pointsGrid.Add(row2Points);
        pointsGrid.Add(row3Points);

        ResetControlPoints();
        PetalControlPoints();
        GenerateVertices();  
    }

    // Update is called once per frame
    void Update()
    {
        // RenderVertices();
        if (Input.GetKeyDown("r")) {
            ResetControlPoints();
            PetalControlPoints();
            GenerateVertices();
        }
    }

    // Source: https://stackoverflow.com/questions/12983731/algorithm-for-calculating-binomial-coefficient
    long BinomialCoefficient(int N, int K)
    {
        long r = 1;
        long d;
        if (K > N) return 0;
        for (d = 1; d <= K; d++)
        {
            r *= N--;
            r /= d;
        }
        return r;
    }

    float B(int n, int i, float u) {
        long nCi = BinomialCoefficient(n, i);
        float a = Mathf.Pow(u, (float)i);
        float b = Mathf.Pow(1.0f - u, (float)(n - i));
        return nCi * a * b; 
    }

    // Get a point on the bicubic patch
    // s and t are floats between 0 and 1
    // Idea for using bicubic patches: http://algorithmicbotany.org/papers/abop/abop.pdf page 119
    // Implementing bicubic patches: https://en.wikipedia.org/wiki/Bézier_surface

    Vector3 GetPoint(float u, float v) 
    {
        Vector3 output = new Vector3(0, 0, 0);
        int n = pointsGrid.Count - 1;
        int m = pointsGrid[0].Count - 1;
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= m; j++) {
                output = output + (B(n, i, u) * B(m, j, v) * pointsGrid[i][j].transform.position);
            }
        }
        return output;
    }


    public void GenerateVertices() 
    {
        vertices = new List<List<Vector3>>();

        for (int i = 0; i <= num_vertices_s; i++) {
            List<Vector3> verticesRow = new List<Vector3>();
            for (int j = 0; j <= num_vertices_t; j++) {
                Vector3 vertex = GetPoint(((float)i) / num_vertices_s, ((float)j) / num_vertices_t);
                verticesRow.Add(vertex);
            }
            vertices.Add(verticesRow);
        }
    }

    void RenderVertices() {
        for (int i = 0; i < num_vertices_s; i++) {
            for (int j = 0; j < num_vertices_t; j++) {
                vertexMarkers[i][j].transform.position = vertices[i][j];
                vertexMarkers[i][j].SetActive(showVertices);
            }
        }
    }

    public List<List<Vector3>> GetVertices() {
        return vertices;
    }

}
