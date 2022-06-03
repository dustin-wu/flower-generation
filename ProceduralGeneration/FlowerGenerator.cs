using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text;

// Positions petals around a central axis to generate flowers

public class FlowerGenerator : MonoBehaviour
{
    public GameObject organ;
    public GameObject petal;
    public Petal petalGenerator;
    Vector3 origin = new Vector3(0, 0, 0.5f);
    List<GameObject> generatedPetals;
    public int petalCount;
    public float c;
    public float basePitch;
    public float pitchRate;

    public GameObject mergedMesh;

    void Change() 
    {
        petalGenerator.ResetControlPoints();
        petalGenerator.PetalControlPoints();
        petalGenerator.GenerateVertices();
        petalCount = Random.Range(12, 53);
        c = Random.Range(0.01f, 0.05f);
        basePitch = Random.Range(25f, 45f);
        pitchRate = Random.Range(0.5f, 2f);
        pitchRate = pitchRate * (Random.Range(0, 2) * 2 - 1);

        // Debug.Log("petalCount: " + petalCount);
        // Debug.Log("c: " + c);
        // Debug.Log("basePitch: " + basePitch);
        // Debug.Log("pitchRate: " + pitchRate);
    }

    void Generate()
    {
        foreach(GameObject p in generatedPetals) {
            Destroy(p);
        }
        generatedPetals = new List<GameObject>();

        

        // earlier petals are higher up and closer to the center
        for (int n = 0; n < petalCount; n++) {
            float phi = n * 137.5f;
            float r = c * Mathf.Sqrt(n);

            GameObject newPetal = Instantiate(petal, new Vector3(0, 0, 0), Quaternion.identity);
            Vector3 petalCenter = newPetal.transform.GetComponent<Renderer>().bounds.center;
            newPetal.transform.RotateAround(origin, newPetal.transform.up, phi);
            newPetal.transform.RotateAround(origin, newPetal.transform.forward, basePitch + n * pitchRate);
            newPetal.transform.position += newPetal.transform.right * r;
            generatedPetals.Add(newPetal);
        }

        CombineInstance[] combine = new CombineInstance[petalCount];
        for (int n = 0; n < petalCount; n++) {
            combine[n].mesh = generatedPetals[n].transform.GetComponent<MeshFilter>().sharedMesh;
            combine[n].transform = generatedPetals[n].transform.localToWorldMatrix;
            generatedPetals[n].gameObject.SetActive(false);
        }
        mergedMesh.transform.GetComponent<MeshFilter>().mesh = new Mesh();
        mergedMesh.transform.GetComponent<MeshFilter>().mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mergedMesh.transform.GetComponent<MeshFilter>().mesh.CombineMeshes(combine);
        mergedMesh.transform.gameObject.SetActive(true);
    }

    // Start is called before the first frame update
    void Start()
    {
        generatedPetals = new List<GameObject>();
        Generate();
        StartCoroutine(ExportFlowers(5000));   
        
    }

    IEnumerator ExportFlowers(int num) {
        for (int i = 0; i < num; i++) {
            yield return new WaitForSeconds(0.001f);
            Change();
            Generate();
            ExportAMesh(mergedMesh.transform.GetComponent<MeshFilter>(), "/Users/dustinwu/Downloads/Flowers/flower" + i.ToString() + ".obj");
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown("r")) {
            Change();
            Generate();
        }
        if (Input.GetKeyDown("f")) {
            Generate();
        }
        if (Input.GetKeyDown("s")) {
            ExportAMesh(mergedMesh.transform.GetComponent<MeshFilter>(), "/Users/dustinwu/Downloads/test.obj");
        }
    }
    Vector3 RotateAroundPoint(Vector3 point, Vector3 pivot, Quaternion angle)
    {
        return angle * (point - pivot) + pivot;
    }
    Vector3 MultiplyVec3s(Vector3 v1, Vector3 v2)
    {
        return new Vector3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
    }
    private string ConstructOBJString(int index)
    {
        string idxString = index.ToString();
        return idxString + "/" + idxString + "/" + idxString;
    }

    void ExportAMesh(MeshFilter meshFilter, string exportPath) {
        //init stuff
        Dictionary<string, bool> materialCache = new Dictionary<string, bool>();
        var exportFileInfo = new System.IO.FileInfo(exportPath);
        string baseFileName = System.IO.Path.GetFileNameWithoutExtension(exportPath);
        //EditorUtility.DisplayProgressBar("Exporting OBJ", "Please wait.. Starting export.", 0);

        //get list of required export things
        MeshFilter[] sceneMeshes;
        List<MeshFilter> tempMFList = new List<MeshFilter>();
        tempMFList.Add(meshFilter);
        sceneMeshes = tempMFList.ToArray();
        
        //work on export
        StringBuilder sb = new StringBuilder();
        StringBuilder sbMaterials = new StringBuilder();
        sb.AppendLine("# Export of " + Application.loadedLevelName);

        bool applyScale = true;
        bool applyRotation = true;
        bool applyPosition = true;

        float maxExportProgress = (float)(sceneMeshes.Length + 1);
        int lastIndex = 0;
        for(int i = 0; i < sceneMeshes.Length; i++)
        {
            string meshName = sceneMeshes[i].gameObject.name;
            float progress = (float)(i + 1) / maxExportProgress;
            MeshFilter mf = sceneMeshes[i];
            MeshRenderer mr = sceneMeshes[i].gameObject.GetComponent<MeshRenderer>();


            //export the meshhh :3
            Mesh msh = mf.sharedMesh;
            int faceOrder = (int)Mathf.Clamp((mf.gameObject.transform.lossyScale.x * mf.gameObject.transform.lossyScale.z), -1, 1);
            
            //export vector data (FUN :D)!
            foreach (Vector3 vx in msh.vertices)
            {
                Vector3 v = vx;
                if (applyScale)
                {
                    v = MultiplyVec3s(v, mf.gameObject.transform.lossyScale);
                }
                
                if (applyRotation)
                {
  
                    v = RotateAroundPoint(v, Vector3.zero, mf.gameObject.transform.rotation);
                }

                if (applyPosition)
                {
                    v += mf.gameObject.transform.position;
                }
                v.x *= -1;
                sb.AppendLine("v " + v.x + " " + v.y + " " + v.z);
            }
            foreach (Vector3 vx in msh.normals)
            {
                Vector3 v = vx;
                
                if (applyScale)
                {
                    v = MultiplyVec3s(v, mf.gameObject.transform.lossyScale.normalized);
                }
                if (applyRotation)
                {
                    v = RotateAroundPoint(v, Vector3.zero, mf.gameObject.transform.rotation);
                }
                v.x *= -1;
                sb.AppendLine("vn " + v.x + " " + v.y + " " + v.z);

            }
            foreach (Vector2 v in msh.uv)
            {
                sb.AppendLine("vt " + v.x + " " + v.y);
            }

            for (int j=0; j < msh.subMeshCount; j++)
            {
                if(mr != null && j < mr.sharedMaterials.Length)
                {
                    string matName = mr.sharedMaterials[j].name;
                    sb.AppendLine("usemtl " + matName);
                }
                else
                {
                    sb.AppendLine("usemtl " + meshName + "_sm" + j);
                }

                int[] tris = msh.GetTriangles(j);
                for(int t = 0; t < tris.Length; t+= 3)
                {
                    int idx2 = tris[t] + 1 + lastIndex;
                    int idx1 = tris[t + 1] + 1 + lastIndex;
                    int idx0 = tris[t + 2] + 1 + lastIndex;
                    if(faceOrder < 0)
                    {
                        sb.AppendLine("f " + ConstructOBJString(idx2) + " " + ConstructOBJString(idx1) + " " + ConstructOBJString(idx0));
                    }
                    else
                    {
                        sb.AppendLine("f " + ConstructOBJString(idx0) + " " + ConstructOBJString(idx1) + " " + ConstructOBJString(idx2));
                    }
                    
                }
            }

            lastIndex += msh.vertices.Length;
        }

        //write to disk
        System.IO.File.WriteAllText(exportPath, sb.ToString());
        Debug.Log("Export to " + exportPath + " was successful");
    }
}
