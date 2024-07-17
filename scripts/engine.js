var {mat4,vec3} = glMatrix



async function main() {

  const canvas = document.querySelector("#canvas")
  canvas.width = window.innerWidth
  canvas.height = window.innerHeight

  const gl = canvas.getContext("webgl2")
  if (!gl) {
    return;
  }



  const skinProgramInfo = CreateShader(gl, skinVS, fs);
  const meshProgramInfo = CreateShader(gl, meshVS, fs);

  class Skin {
    constructor(joints, inverseBindMatrixData) {
      this.joints = joints;
      this.inverseBindMatrices = [];
      this.jointMatrices = [];
      this.jointData = new Float32Array(joints.length * 16);
      for (let i = 0; i < joints.length; ++i) {
        this.inverseBindMatrices.push(new Float32Array(
            inverseBindMatrixData.buffer,
            inverseBindMatrixData.byteOffset + Float32Array.BYTES_PER_ELEMENT * 16 * i,
            16));
        this.jointMatrices.push(new Float32Array(
            this.jointData.buffer,
            Float32Array.BYTES_PER_ELEMENT * 16 * i,
            16));
      }
      this.jointTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.jointTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    }
    update(node) {
      let globalWorldInverse = mat4.create()
      mat4.invert(globalWorldInverse, node.worldMatrix);
      for (let j = 0; j < this.joints.length; ++j) {
        const joint = this.joints[j];
        const dst = this.jointMatrices[j];
        mat4.multiply(dst, joint.worldMatrix, globalWorldInverse)
        mat4.multiply(dst, dst, this.inverseBindMatrices[j])
      }
      gl.bindTexture(gl.TEXTURE_2D, this.jointTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 4, this.joints.length, 0,
                    gl.RGBA, gl.FLOAT, this.jointData);
    }
  }

  class TRS {
    constructor(position = [0, 0, 0], rotation = [0, 0, 0, 1], scale = [1, 1, 1]) {
      this.position = position;
      this.rotation = rotation;
      this.scale = scale;
    }
    getMatrix(dst) {
      dst = dst || new Float32Array(16);
      mat4.fromRotationTranslationScale(dst, this.rotation, this.position, this.scale)
      return dst;
    }
  }

  class Node {
    constructor(source, name) {
      this.name = name;
      this.source = source;
      this.parent = null;
      this.children = [];
      this.localMatrix = mat4.create()
      mat4.identity(this.localMatrix);
      this.worldMatrix = mat4.create()
      mat4.identity(this.worldMatrix);
      this.drawables = [];
    }
    setParent(parent) {
      if (this.parent) {
        this.parent._removeChild(this);
        this.parent = null;
      }
      if (parent) {
        parent._addChild(this);
        this.parent = parent;
      }
    }
    updateWorldMatrix(parentWorldMatrix) {
      const source = this.source;
      if (source) {
        source.getMatrix(this.localMatrix);
      }

      if (parentWorldMatrix) {
        mat4.multiply(this.worldMatrix, parentWorldMatrix, this.localMatrix);
      } else {
        mat4.copy(this.worldMatrix, this.localMatrix)
      }

      // now process all the children
      const worldMatrix = this.worldMatrix;
      for (const child of this.children) {
        child.updateWorldMatrix(worldMatrix);
      }
    }
    traverse(fn) {
      fn(this);
      for (const child of this.children) {
        child.traverse(fn);
      }
    }
    _addChild(child) {
      this.children.push(child);
    }
    _removeChild(child) {
      const ndx = this.children.indexOf(child);
      this.children.splice(ndx, 1);
    }
  }

  class SkinRenderer {
    constructor(mesh, skin) {
      this.mesh = mesh;
      this.skin = skin;
    }
    render(node, projection, view, sharedUniforms) {
      const {skin, mesh} = this;
      skin.update(node);
      gl.useProgram(skinProgramInfo.program);
      for (const primitive of mesh.primitives) {
        gl.bindVertexArray(primitive.vao);
        var uni = uniformsSetter(gl,skinProgramInfo)
        uni.set({
          u_projection: projection,
          u_view: view,
          u_world: node.worldMatrix,
          u_jointTexture: skin.jointTexture,
          u_numJoints: skin.joints.length,
        })
        uni.set(primitive.material.uniforms)
        uni.set(sharedUniforms)
        gl.drawElements(gl.TRIANGLES, primitive.bufferInfo.numElements, gl.UNSIGNED_SHORT, 0);
      }
    }
  }

  class MeshRenderer {
    constructor(mesh) {
      this.mesh = mesh;
    }
    render(node, projection, view, sharedUniforms) {
      const {mesh} = this;
      gl.useProgram(meshProgramInfo.program);
      for (const primitive of mesh.primitives) {
        gl.bindVertexArray(primitive.vao);
        var uni = uniformsSetter(gl,skinProgramInfo)
        uni.set({
          u_projection: projection,
          u_view: view,
          u_world: node.worldMatrix,
        })
        uni.set(primitive.material.uniforms)
        uni.set(sharedUniforms)
        gl.drawElements(gl.TRIANGLES, primitive.bufferInfo.numElements, gl.UNSIGNED_SHORT, 0);
      }
    }
  }

  function throwNoKey(key) {
    throw new Error(`no key: ${key}`);
  }
  function NumSize(val){
    return {
        'SCALAR': 1,
        'VEC2': 2,
        'VEC3': 3,
        'VEC4': 4,
        'MAT2': 4,
        'MAT3': 9,
        'MAT4': 16
    }[val]
}

  const accessorTypeToNumComponentsMap = {
    'SCALAR': 1,
    'VEC2': 2,
    'VEC3': 3,
    'VEC4': 4,
    'MAT2': 4,
    'MAT3': 9,
    'MAT4': 16,
  };

  function accessorTypeToNumComponents(type) {
    return accessorTypeToNumComponentsMap[type] || throwNoKey(type);
  }
  function glTypedArray(val){
    return {
        '5120': Int8Array,
        '5121': Uint8Array,
        '5122': Int16Array,
        '5123': Uint16Array,
        '5124': Int32Array,
        '5125': Uint32Array,
        '5126': Float32Array
    }[val]
}

  // Given an accessor index return a WebGLBuffer and a stride
  function getAccessorAndWebGLBuffer(gl, gltf, accessorIndex) {
    const accessor = gltf.accessors[accessorIndex];
    const bufferView = gltf.bufferViews[accessor.bufferView];
    if (!bufferView.webglBuffer) {
      const buffer = gl.createBuffer();
      const target = bufferView.target || gl.ARRAY_BUFFER;
      const arrayBuffer = gltf.buffers[bufferView.buffer];
      const data = new Uint8Array(arrayBuffer, bufferView.byteOffset, bufferView.byteLength);
      gl.bindBuffer(target, buffer);
      gl.bufferData(target, data, gl.STATIC_DRAW);
      bufferView.webglBuffer = buffer;
    }
    return {
      accessor,
      buffer: bufferView.webglBuffer,
      stride: bufferView.stride || 0,
    };
  }

  function uniformsSetter(gl,shader){
       
    let sampler=0
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D,null)
  
    return {
        set(uniforms){

          if(!uniforms){
            return
          }
            
            for(const key of Object.keys(uniforms)){

                const pointer = gl.getUniformLocation(shader.program,key)
                const Value=uniforms[key]
            
                if((! pointer)||(Value===undefined)){
                    return
                }
                
                
                
                if((Value!==undefined)&&(Value["constructor"]["name"]==="WebGLTexture")){
                    gl.uniform1i(pointer,sampler)
                    gl.activeTexture(gl.TEXTURE0+sampler)
                    gl.bindTexture(gl.TEXTURE_2D,Value)
                    sampler++
                }else{
                    
                    switch(Value.length) {
                        case 16:
                            gl.uniformMatrix4fv(pointer,null,Value)
                            break;
                        case 4:
                            gl.uniform4fv(pointer,Value)
                            break;
                        case 3:
                            gl.uniform3fv(pointer,Value)
                            break;
                        case 2:
                            gl.uniform2fv(pointer,Value)
                            break;
                        default:
                            gl.uniform1f(pointer,Value)
                      }
                    

                }
        
            }
            
        }
    }
}

  async function loadGLTF(url) {
    const gltf = await loadJSON(url);

    // load all the referenced files relative to the gltf file
    const baseURL = new URL(url, location.href);
    gltf.buffers = await Promise.all(gltf.buffers.map((buffer) => {
      const url = new URL(buffer.uri, baseURL.href);
      return loadBinary(url.href);
    }));


    gltf['data']=gltf['accessors'].map(accessor=>{
      const bufferView=gltf['bufferViews'][accessor['bufferView']]
      const TypedArray = glTypedArray(accessor['componentType'])
      return new TypedArray(
          gltf.buffers[bufferView['buffer']],
          bufferView['byteOffset'],//+ (accessor.byteOffset || 0),
          accessor['count']* NumSize(accessor['type']))
  })

    const defaultMaterial = {
      uniforms: {
        u_diffuse: [.5, .8, 1, 1],
      },
    };

    // setup meshes
    gltf.meshes.forEach((mesh) => {
      mesh.primitives.forEach((primitive) => {
        var vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        const attribs = {};
        let numElements;
        for (const [attribName, index] of Object.entries(primitive.attributes)) {
          const {accessor, buffer, stride} = getAccessorAndWebGLBuffer(gl, gltf, index);

          let key = 'a_'+attribName
          const attribute = gl.getAttribLocation(skinProgramInfo.program, key)
          if(attribute>-1){
            console.log(key)
            if(key=='a_JOINTS_0'){
              gl.vertexAttribIPointer(attribute, accessorTypeToNumComponents(accessor.type), 
              gl.UNSIGNED_SHORT, false,0,0)
            }else{
              gl.vertexAttribPointer(attribute, accessorTypeToNumComponents(accessor.type), 
                gl.FLOAT, false,0,0)
            }
            gl.enableVertexAttribArray(attribute)
          }

          numElements = accessor.count;
          attribs[`a_${attribName}`] = {
            buffer,
            type: accessor.componentType,
            numComponents: accessorTypeToNumComponents(accessor.type),
            stride,
            offset: accessor.byteOffset | 0,
          };
        }

        const bufferInfo = {
          attribs,
          numElements,
        };

        if (primitive.indices !== undefined) {
          const {accessor, buffer} = getAccessorAndWebGLBuffer(gl, gltf, primitive.indices);
          bufferInfo.numElements = accessor.count;
          bufferInfo.indices = buffer;
          bufferInfo.elementType = accessor.componentType;
        }

        primitive.bufferInfo = bufferInfo;

        primitive.vao = vao//twgl.createVAOFromBufferInfo(gl, skinProgramInfo, primitive.bufferInfo);
        //primitive.vao = createVAOFromBufferInfo(gl, skinProgramInfo, primitive.bufferInfo);

        // save the material info for this primitive
        primitive.material = gltf.materials && gltf.materials[primitive.material] || defaultMaterial;
      });
    });

    const skinNodes = [];
    const origNodes = gltf.nodes;
    gltf.nodes = gltf.nodes.map((n) => {
      const {name, skin, mesh, translation, rotation, scale} = n;
      const trs = new TRS(translation, rotation, scale);
      const node = new Node(trs, name);
      const realMesh = gltf.meshes[mesh];
      if (skin !== undefined) {
        skinNodes.push({node, mesh: realMesh, skinNdx: skin});
      } else if (realMesh) {
        node.drawables.push(new MeshRenderer(realMesh));
      }
      return node;
    });

    // setup skins
    gltf.skins = gltf.skins.map((skin) => {
      const joints = skin.joints.map(ndx => gltf.nodes[ndx]);
      //const {stride, array} = gltf.data[skin.inverseBindMatrices]//getAccessorTypedArrayAndStride(gl, gltf, skin.inverseBindMatrices);
      return new Skin(joints, gltf.data[skin.inverseBindMatrices]);
    });

    // Add SkinRenderers to nodes with skins
    for (const {node, mesh, skinNdx} of skinNodes) {
      node.drawables.push(new SkinRenderer(mesh, gltf.skins[skinNdx]));
    }

    // arrange nodes into graph
    gltf.nodes.forEach((node, ndx) => {
      const children = origNodes[ndx].children;
      if (children) {
        addChildren(gltf.nodes, node, children);
      }
    });

    // setup scenes
    for (const scene of gltf.scenes) {
      scene.root = new Node(new TRS(), scene.name);
      addChildren(gltf.nodes, scene.root, scene.nodes);
    }

    console.log('gltf',gltf)

    return gltf;
  }

  function addChildren(nodes, node, childIndices) {
    childIndices.forEach((childNdx) => {
      const child = nodes[childNdx];
      child.setParent(node);
    });
  }

  async function loadFile(url, typeFunc) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`could not load: ${url}`);
    }
    return await response[typeFunc]();
  }

  async function loadBinary(url) {
    return loadFile(url, 'arrayBuffer');
  }

  async function loadJSON(url) {
    return loadFile(url, 'json');
  }

  const gltf = await loadGLTF('/models/whale/whale.CYCLES.gltf');
  //const gltf = await loadGLTF('/models/stickman/scene.gltf');
  //const gltf = await loadGLTF('/models/henry_stickmin/scene.gltf');
  

  function degToRad(deg) {
    return deg * Math.PI / 180;
  }




  function lerp(input, target, percent) {
    input += (target - input)*percent;
    return input
}
function lerpVec(input, target, percent) {
    return input.map((p,i)=>lerp(input[i], target[i], percent))
}






  //const origMatrices = new Map();
  function animSkin(model, skin, a) {

    if(!model.animData){
      model.animData = {frames: 15}
  }


  if(!model['animations']||!model.animations.length){
      return
  }

      const animation=model.animations[0]



      model.animData.frame = model.animData.frame??0
      model.animData.percent=model.animData.percent??0.0;
      model.animData.frames=model.animData.frames??0
      model.animData.time=model.animData.time??0

  
      const speed = 0.1

      model.animData.time+=a//delta

    
      model.animData.percent = model.animData.time/speed

      if(model.animData.frame>10){
        model.animData.frame=0
        model.animData.time=0
      }

      

    for(const channel of animation['channels']){
      const target=channel['target']
      const joint=model.nodes[target['node']]
      const sampler=animation['samplers'][channel['sampler']]
      
      const fname={'translation':'position','rotation':'rotation','scale':'scale',}[channel.target['path']]
      const len=(fname==='rotation')?4:3

     
      const accessor=model.data[sampler['output']]
      
      if(!model.animData.frames){
          model.animData.frames=accessor.length/len
      }
  
      if(fname==='rotation'){
          var fromVal=[accessor[(model.animData.frame*len)+0],accessor[(model.animData.frame*len)+1],accessor[(model.animData.frame*len)+2],accessor[(model.animData.frame*len)+3]]
      }else{
          var fromVal=[accessor[(model.animData.frame*len)+0],accessor[(model.animData.frame*len)+1],accessor[(model.animData.frame*len)+2]]
      }

      model.animData.frame++
      if(fname==='rotation'){
        var value=[accessor[(model.animData.frame*len)+0],accessor[(model.animData.frame*len)+1],accessor[(model.animData.frame*len)+2],accessor[(model.animData.frame*len)+3]]
      }else{
          var value=[accessor[(model.animData.frame*len)+0],accessor[(model.animData.frame*len)+1],accessor[(model.animData.frame*len)+2]]
      }
      model.animData.frame--

  
      const newVal=lerpVec(fromVal, value, model.animData.percent)

      joint.source[fname]=newVal
      }

      while(model.animData.time>speed){
        model.animData.time-=speed
        model.animData.frame++
      }
      
  }






  let delta = 0
  let lastTime = 0
  function render(time) {
    time *= 0.001;
    delta = time-lastTime
    lastTime = time

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.clearColor(.5, .5, .5, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const fieldOfViewRadians = degToRad(60);
    const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
    const projection = mat4.create()
    mat4.perspective(projection, fieldOfViewRadians, aspect, 1, 20000);

    const cameraPosition = [0, 0, -200];
    const target = [0, 0, 0];
    // for debugging .. see article
    // const cameraPosition = [5, 0, 5];
    // const target = [0, 0, 0];
    const up = [0, 1, 0];
    // Compute the camera's matrix using look at.
    const camera = mat4.create()
    mat4.lookAt(camera, cameraPosition, target, up);

    // Make a view matrix from the camera matrix.
    let view = mat4.create()
    mat4.invert(view, camera);

    view = mat4.create()
    view[14] = -200


    animSkin(gltf, gltf.skins[0], delta);

    let outVec = vec3.create()
    const sharedUniforms = {
      u_lightDirection: vec3.normalize(outVec,[-1, 3, 5]),
    };

    function renderDrawables(node) {
      for (const drawable of node.drawables) {
        drawable.render(node, projection, view, sharedUniforms);
      }
    }

    for (const scene of gltf.scenes) {
      // updatte all world matices in the scene.
      scene.root.updateWorldMatrix();
      // walk the scene and render all renderables
      scene.root.traverse(renderDrawables);
    }

    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}

main();
