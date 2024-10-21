Shader "SlapSplat_G/SlapSplat_G_T" {
Properties {
	_GeoData ("Geo Data", 2D) = "white" {}
	_OrderData ("Order Data", 2D) = "white" {}
	_CustomScale( "Custom Scale", float) = 1.0
	_Gamma ( "gamma", float ) = 1.0
	_EmissionAmount( "emission amount", float ) = 0.16
	_Brightness( "brightness", float ) = 1.0
	[Toggle(_UseSH)] _UseSH( "Use SH", int ) = 0
	[Toggle(_SHLocal)] _SHLocal ("SH Per Pixel", int) = 0
	_SHData ("SH Data", 2D) = "white" {}
	_SHImpact ("SH Impact", float ) = 0.125
}
SubShader {	

	Pass { 
	Tags { "RenderType"="Transparent" "Queue"="Transparent"}
		Cull Off
		Blend SrcAlpha OneMinusSrcAlpha
		CGPROGRAM

		#pragma multi_compile_local _ _UseSH
		#pragma multi_compile_local _ _SHLocal
		
		#include "slapsplat.cginc"
		
		#pragma vertex VertexProgram
		#pragma geometry GeometryProgram
		#pragma fragment frag earlydepthstencil
		#pragma target 5.0
		#pragma multi_compile_fwdadd_fullshadows

		float4 frag(g2f i ) : SV_Target
		{
			UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX( i );
			
			float fakeAlpha;
			float squircleness = 2.0;
			float2 tc = abs( i.tc.xy );
			float inten = 1.1 - ( pow( tc.x, squircleness ) + pow( tc.y, squircleness ) ) / 1.1;
			
			fakeAlpha=inten;

			float3 color;
			clip( fakeAlpha - 0.5 );
			color = pow( i.color.rgb, 1.4 );
			float4 clipPos = mul(UNITY_MATRIX_VP, i.hitworld );
			
			color *= _LightColor0 + i.emission.rgb; 
			
			return max( float4( color, fakeAlpha ), 0 );
		}
		ENDCG
	}
}
}