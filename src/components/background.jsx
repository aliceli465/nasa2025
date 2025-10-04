import gsap from "gsap";
import { useRef, useEffect } from "react";
import Button from "./Button";
import { TiLocationArrow } from "react-icons/ti";
import { FaSun, FaMoon } from "react-icons/fa";

const Background = () => {
  const sunRef = useRef(null);
  const shadowRef = useRef(null);
  const orbitLineRef = useRef(null);

  useEffect(() => {
    const tl = gsap.timeline({ repeat: -1 });

    // Orbit animation
    tl.to(orbitLineRef.current, {
      strokeDashoffset: 0,
      duration: 8,
      ease: "none",
    });

    // Phase highlighting
    gsap
      .timeline({ repeat: -1, repeatDelay: 1 })
      .to(sunRef.current, {
        scale: 1.05,
        opacity: 1,
        duration: 1.5,
        ease: "power2.inOut",
      })
      .to(sunRef.current, {
        scale: 1,
        opacity: 0.6,
        duration: 0.5,
      })
      .to(shadowRef.current, {
        scale: 1.05,
        opacity: 1,
        duration: 1.5,
        ease: "power2.inOut",
      })
      .to(shadowRef.current, {
        scale: 1,
        opacity: 0.6,
        duration: 0.5,
      });
  }, []);

  return (
    <div className="min-h-screen w-full bg-black text-[#dfdff2] py-20 px-5">
      <div className="max-w-7xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <p className="font-circular-web text-lg text-[#dfdff2]/80 max-w-2xl mx-auto leading-relaxed">
            When a satellite is in orbit, it spends half of its time in the sun
            and the other half in Earth's shadow, where temperatures drop to
            -150°C
          </p>
          <p className="font-circular-web text-lg text-[#dfdff2]/80 max-w-2xl mx-auto leading-relaxed mt-8">
            This means we have infinite solar energy and infinite cooling. Zero
            cooling infrastructure needed.
          </p>
        </div>

        {/* Visual Orbit Cycle */}
        <div className="relative mb-20">
          <div className="flex items-center justify-center gap-20 md:gap-32">
            {/* Orbit Line */}
            <svg
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-3xl mt-8 z-0"
              viewBox="0 0 600 150"
            >
              <ellipse
                ref={orbitLineRef}
                cx="300"
                cy="75"
                rx="250"
                ry="60"
                fill="none"
                stroke="url(#gradient)"
                strokeWidth="1.5"
                strokeDasharray="1600"
                strokeDashoffset="1600"
                opacity="0.4"
              />
              <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#fbbf24" />
                  <stop offset="50%" stopColor="#60a5fa" />
                  <stop offset="100%" stopColor="#fbbf24" />
                </linearGradient>
              </defs>
            </svg>

            {/* Sun Phase */}
            <div
              ref={sunRef}
              className="text-center transition-all relative z-10"
            >
              <div className="relative mb-4">
                <FaSun className="text-6xl md:text-7xl text-yellow-400 mx-auto" />
                <div className="absolute inset-0 bg-yellow-400/20 rounded-full blur-2xl"></div>
              </div>
              <h3 className="font-robert-medium text-xl text-yellow-300 mb-2">
                45 min
              </h3>
              <p className="font-circular-web text-[#dfdff2]/70">
                Solar Charging
              </p>
            </div>

            {/* Shadow Phase */}
            <div
              ref={shadowRef}
              className="text-center transition-all relative z-10"
            >
              <div className="relative mb-4">
                <FaMoon className="text-6xl md:text-7xl text-blue-400 mx-auto" />
                <div className="absolute inset-0 bg-blue-400/20 rounded-full blur-2xl"></div>
              </div>
              <h3 className="font-robert-medium text-xl text-blue-300 mb-2">
                45 min
              </h3>
              <p className="font-circular-web text-[#dfdff2]/70">
                Peak Computing
              </p>
            </div>
          </div>

          <p className="text-center mt-20 text-[#dfdff2]/60"></p>
        </div>
        <br></br>
        <div className="text-center mt-20 mb-10">
          <h2 className="font-circular-web text-5xl text-[#dfdff2]/80 max-w-2xl mx-auto leading-relaxed">
            Enter, LeData.
          </h2>
        </div>
      </div>
    </div>
  );
};

export default Background;
