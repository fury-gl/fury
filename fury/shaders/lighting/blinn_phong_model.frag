// Calculate the diffuse factor and diffuse color
df = max(0, normal.z);
diffuse = df * diffuseColor * lightColor0;

// Calculate the specular factor and specular color
sf = pow(df, specularPower);
specular = sf * specularColor * lightColor0;

// Blinn-Phong illumination model
fragOutput0 = vec4(ambientColor + diffuse + specular, opacity);
