import gsap from "gsap";
import { useRef } from "react";

import Button from "./Button";
import AnimatedTitle from "./AnimatedTitle";
import { TiLocationArrow } from "react-icons/ti";

const DemoLink = () => {
  const frameRef = useRef(null);

  const handleMouseMove = (e) => {
    const { clientX, clientY } = e;
    const element = frameRef.current;

    if (!element) return;

    const rect = element.getBoundingClientRect();
    const xPos = clientX - rect.left;
    const yPos = clientY - rect.top;

    const centerX = rect.width / 2;
    const centerY = rect.height / 2;

    const rotateX = ((yPos - centerY) / centerY) * -10;
    const rotateY = ((xPos - centerX) / centerX) * 10;

    gsap.to(element, {
      duration: 0.3,
      rotateX,
      rotateY,
      transformPerspective: 500,
      ease: "power1.inOut",
    });
  };

  const handleMouseLeave = () => {
    const element = frameRef.current;

    if (element) {
      gsap.to(element, {
        duration: 0.3,
        rotateX: 0,
        rotateY: 0,
        ease: "power1.inOut",
      });
    }
  };

  return (
    <div
      id="story"
      className="min-h-dvh w-full bg-black text-[#dfdff2] overflow-hidden"
    >
      <div className="flex size-full flex-col items-center justify-center py-10 pb-24 text-center">
        <AnimatedTitle
          title="Se<b>e</b> our <b>de</b>mo <br /> tod<b>ay</b>"
          containerClass="mt-5 pointer-events-none mix-blend-difference relative z-10"
        />
        <br />
        <p className="mt-3 max-w-sm font-circular-web text-[#dfdff2]">
          Tech giants need massive compute. We're here. The space economy is
          here. Your next breakthrough is in orbit.
        </p>
        <br />
        <Button
          id="product-button"
          title="Code demo"
          rightIcon={<TiLocationArrow />}
          containerClass="bg-[#dfdff2] md:flex hidden items-center justify-center gap-1"
        />
      </div>
    </div>
  );
};

export default DemoLink;
