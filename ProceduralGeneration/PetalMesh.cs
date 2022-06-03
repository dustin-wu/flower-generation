using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PetalMesh : ProcBase
{
    public Petal petal;

    // Builds a bicubic patch to use for generating petals

    public override Mesh BuildMesh()
	{
        MeshBuilder meshBuilder = new MeshBuilder();

        List<List<Vector3>> vertices = petal.GetVertices();

        for (int i = 0; i < vertices.Count; i++) {
            
            float v = (1.0f / vertices.Count) * i;

            for (int j = 0; j < vertices[i].Count; j++) {
                
                float u = (1.0f / vertices[i].Count) * j;
                Vector2 uv = new Vector2(u, v);
                bool buildTriangles = i > 0 && j > 0;
                BuildQuadForGrid(meshBuilder, vertices[i][j], uv, buildTriangles, vertices[i].Count, false);

            }
        }

        for (int i = 0; i < vertices.Count; i++) {
            
            float v = (1.0f / vertices.Count) * i;

            for (int j = 0; j < vertices[i].Count; j++) {
                
                float u = (1.0f / vertices[i].Count) * j;
                Vector2 uv = new Vector2(u, v);
                bool buildTriangles = i > 0 && j > 0;
                BuildQuadForGrid(meshBuilder, vertices[i][j], uv, buildTriangles, vertices[i].Count, true);
            }
        }

        Mesh mesh = meshBuilder.CreateMesh();
		mesh.RecalculateNormals();
		return mesh;
    }
}
