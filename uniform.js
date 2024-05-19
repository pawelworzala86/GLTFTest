function uniformsSetter(gl,shader){
       
        let sampler=0
        gl.activeTexture(gl.TEXTURE0)
        gl.bindTexture(gl.TEXTURE_2D,null)
      
        return {
            set(uniforms){
                
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